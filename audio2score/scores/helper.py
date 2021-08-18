import sys
import music21 as m21
from collections import Counter

from audio2score.scores.music21score import ScoreM21



def split_bars(score_file, N_bars):
    # input type: score_file in xml format
    # return type: List(ScoreM21)
    s = m21.converter.parse(score_file)
    N_measures = len(list(s.parts)[0].getElementsByClass(m21.stream.Measure))
    if N_bars != N_measures:
        s = s.expandRepeats()
    try:
        s = s.expandRepeats()
    except:
        print('no repeat')
        
    right_hand, left_hand = tuple(s.parts) if len(list(s.parts)) == 2 else tuple(list(s.parts)[1:3])
    right_hand, left_hand = right_hand.chordify(), left_hand.chordify()
    measures_right = list(right_hand.getElementsByClass(m21.stream.Measure))
    measures_left = list(left_hand.getElementsByClass(m21.stream.Measure))
    
    score_m21_list = []
    key_signature = m21.key.KeySignature(0)
    time_signature = m21.meter.TimeSignature('4/4')

    for bar_index in range(len(measures_right)):
        k = list(measures_right[bar_index].getElementsByClass(m21.key.KeySignature))
        key_signature = m21.key.KeySignature(k[0].sharps) if k else m21.key.KeySignature(key_signature.sharps)
        t = list(measures_right[bar_index].getElementsByClass(m21.meter.TimeSignature))
        time_signature = t[0] if t else m21.meter.TimeSignature(time_signature.ratioString)
        
        score_m21 = ScoreM21(key_signature, time_signature, measures_right[bar_index], measures_left[bar_index])
        score_m21_list.append(score_m21)

    return score_m21_list
        
    
def combine_bars_from_score_reshaped(key_signatures, time_signatures, right_list, left_list, quarter_lengths):
    # return type: m21.stream.Score
    N_bars = len(key_signatures)
    if len(time_signatures) != N_bars or len(right_list) != N_bars or len(left_list) != N_bars:
        sys.exit('Length Error! number of bars not equal!')
    
    score_m21_list = []
    for bar_index in range(N_bars)[::-1]:
        if bar_index == N_bars - 1:
            score_m21 = ScoreM21(key_signatures[bar_index], time_signatures[bar_index],
                                right_list[bar_index].to_m21(), left_list[bar_index].to_m21())
        else:
            score_m21 = ScoreM21(key_signatures[bar_index], time_signatures[bar_index],
                                right_list[bar_index].to_m21(), left_list[bar_index].to_m21(),
                                next_tied_pitches=score_m21_list[0].next_tied_pitches)
        score_m21_list.insert(0, score_m21)
    
    s = combine_bars(score_m21_list, quarter_lengths)
    s = grammarise_xml_score(s)

    return s


def combine_bars_from_score_lilypond(key_signatures, time_signatures, right_list, left_list, quarter_lengths):
    # return type: m21.stream.Score
    N_bars = len(key_signatures)
    if len(time_signatures) != N_bars or len(right_list) != N_bars or len(left_list) != N_bars:
        sys.exit('Length Error! number of bars not equal!')
        
    score_m21_list = []
    for bar_index in range(N_bars):
        if bar_index == 0:
            score_m21 = ScoreM21(key_signatures[bar_index], time_signatures[bar_index],
                                right_list[bar_index].to_m21(), left_list[bar_index].to_m21())
        else:
            score_m21 = ScoreM21(key_signatures[bar_index], time_signatures[bar_index],
                                right_list[bar_index].to_m21(), left_list[bar_index].to_m21(),
                                prev_tied_pitches=score_m21_list[-1].prev_tied_pitches)
        score_m21_list.append(score_m21)
        
    s = combine_bars(score_m21_list, quarter_lengths)
    s = grammarise_xml_score(s)

    return s


def combine_bars(score_m21_list, quarter_lengths):
    # input type:  List(ScoreM21)
    # return type: m21.stream.Score
    right_hand, left_hand = m21.stream.Part(), m21.stream.Part()
    for bar_index, score_m21 in enumerate(score_m21_list):
        measure_right, measure_left = m21.stream.Measure(), m21.stream.Measure()

        for measure, measure_raw in [(measure_right, score_m21_list[bar_index].right),
                                    (measure_left, score_m21_list[bar_index].left)]:

            if bar_index == 0 or score_m21.key_signature.sharps != score_m21_list[bar_index-1].key_signature.sharps:
                measure.append(m21.key.KeySignature(score_m21.key_signature.sharps))

            for n in measure_raw.notesAndRests:
                if measure.quarterLength + n.quarterLength > quarter_lengths[bar_index]:
                    n.quarterLength = quarter_lengths[bar_index] - measure.quarterLength
                measure.append(n)
                if measure.quarterLength == quarter_lengths[bar_index]:
                    break

            if measure.quarterLength < quarter_lengths[bar_index]:
                measure.append(m21.note.Rest(quarterLength=quarter_lengths[bar_index]-measure.quarterLength))

        right_hand.append(measure_right)
        left_hand.append(measure_left)

    s = m21.stream.Score()
    s.append([right_hand, left_hand])
    # clefs
    s[0][0].insert(0, m21.clef.TrebleClef())
    s[1][0].insert(0, m21.clef.BassClef())
    return s


def evenup_bars(s):
    right_hand, left_hand = tuple(s.parts)
    measures_right = list(right_hand.getElementsByClass(m21.stream.Measure))
    measures_left = list(left_hand.getElementsByClass(m21.stream.Measure))
    N_bars = len(measures_right)
    
    quarterLength = [m.quarterLength for m in measures_left]
    quarterLength = Counter(quarterLength).most_common(1)[0][0]
    
    s_new = m21.stream.Score()
    s_new.append([m21.stream.Part(0), m21.stream.Part(1)])
    
    for bar_index in range(N_bars):
        measure_new_right = m21.stream.Measure(bar_index)
        measure_new_left = m21.stream.Measure(bar_index)
        
        key_signature = measures_right[bar_index].getElementsByClass(m21.key.KeySignature)
        if key_signature:
            measure_new_right.append(m21.key.KeySignature(key_signature[0].sharps))
            measure_new_left.append(m21.key.KeySignature(key_signature[0].sharps))
        
        for n in measures_right[bar_index].notesAndRests:
            if measure_new_right.quarterLength + n.quarterLength > quarterLength:
                n.quarterLength = quarterLength - measure_new_right.quarterLength
            measure_new_right.append(n)
            if measure_new_right.quarterLength == quarterLength:
                break
        for n in measures_left[bar_index].notesAndRests:
            if measure_new_left.quarterLength + n.quarterLength > quarterLength:
                n.quarterLength = quarterLength - measure_new_left.quarterLength
            measure_new_left.append(n)
            if measure_new_left.quarterLength == quarterLength:
                break
        if measure_new_right.quarterLength < quarterLength:
            measure_new_right.append(m21.note.Rest(quarterLength=quarterLength-measure_new_right.quarterLength))
        if measure_new_left.quarterLength < quarterLength:
            measure_new_left.append(m21.note.Rest(quarterLength=quarterLength-measure_new_left.quarterLength))
            
        s_new[0].append(measure_new_right)
        s_new[1].append(measure_new_left)
            
    # clefs
    s_new[0][0].insert(0, m21.clef.TrebleClef())
    s_new[1][0].insert(0, m21.clef.BassClef())
            
    return s_new
    
def grammarise_xml_score(s):
    right_hand, left_hand = right_hand, left_hand = tuple(s.parts) \
                                                if len(list(s.parts)) == 2 \
                                                else tuple(list(s.parts)[1:3])
    right_hand, left_hand = right_hand.chordify(), left_hand.chordify()
    
    ns_right = list(right_hand.flat.notesAndRests)
    ns_left = list(left_hand.flat.notesAndRests)
    pitches_right = [n.pitches if type(n) != m21.note.Rest else tuple() for n in ns_right]
    pitches_left = [n.pitches if type(n) != m21.note.Rest else tuple() for n in ns_left]
    midi_pitches_right = [set([p.midi for p in pitches]) for pitches in pitches_right]
    midi_pitches_left = [set([p.midi for p in pitches]) for pitches in pitches_left]
    
    # remove duplicate notes
    for ns, midi_pitches in [(ns_right, midi_pitches_right), (ns_left, midi_pitches_left)]:
        for ni, n in enumerate(ns):
            if type(n) == m21.note.Rest:
                continue
            current_midi_pitches = midi_pitches[ni].copy()
            new_notes = []
            raw_notes = tuple([n]) if type(n) == m21.note.Note else n.notes
            for note in raw_notes:
                if note.pitch.midi in current_midi_pitches:
                    new_notes.append(note)
                    current_midi_pitches.remove(note.pitch.midi)
            if len(new_notes) == 1:
                ns[ni] = new_notes[0]
            else:
                ns[ni].notes = tuple(new_notes)
    
    # remove extra ties
    for ns, midi_pitches in [(ns_right, midi_pitches_right), (ns_left, midi_pitches_left)]:
        for ni, n in enumerate(ns):
            if type(n) == m21.note.Rest:
                continue
            prev_midi_pitches = set() if ni == 0 else midi_pitches[ni-1].copy()
            next_midi_pitches = set() if ni == len(ns)-1 else midi_pitches[ni+1].copy()
            notes = tuple([n]) if type(n) == m21.note.Note else n.notes
            for note in notes:
                if note.tie:
                    if note.tie.type == 'start':
                        if note.pitch.midi not in next_midi_pitches:
                            note.tie = None
                    elif note.tie.type == 'end':
                        if note.pitch.midi not in prev_midi_pitches:
                            note.tie = None
                    elif note.tie.type == 'continue':
                        if note.pitch.midi not in prev_midi_pitches and note.pitch.midi not in next_midi_pitches:
                            note.tie = None
                        elif note.pitch.midi not in prev_midi_pitches:
                            note.tie.type = 'start'
                        elif note.pitch.midi not in next_midi_pitches:
                            note.tie.type = 'end'
            if len(notes) == 1:
                ns[ni] = notes[0]
            else:
                ns[ni].notes = tuple(notes)
    
    s_new = m21.stream.Score()
    s_new.append([right_hand, left_hand])
    return s_new
                
                
def remove_tempos(s):
    right_hand, left_hand = tuple(s.parts)
    right_hand, left_hand = right_hand.chordify(), left_hand.chordify()
    measures_right = list(right_hand.getElementsByClass(m21.stream.Measure))
    measures_left = list(left_hand.getElementsByClass(m21.stream.Measure))
    N_bars = len(measures_right)
    
    measures_right_new, measures_left_new = [], []
    for measures_new, measures in [(measures_right_new, measures_right), (measures_left_new, measures_left)]:
        for bar_index in range(N_bars):
            measure_new = m21.stream.Measure()

            k = list(measures[bar_index].getElementsByClass(m21.key.KeySignature))
            if len(k) > 0:
                measure_new.append(m21.key.KeySignature(k[0].sharps))

            t = list(measures[bar_index].getElementsByClass(m21.meter.TimeSignature))
            if len(t) > 0:
                measure_new.append(m21.meter.TimeSignature(t[0].ratioString))

            for n in measures[bar_index].notesAndRests:
                if type(n) == m21.note.Rest:
                    measure_new.append(m21.note.Rest(quarterLength=n.quarterLength))
                elif type(n) == m21.chord.Chord:
                    c = m21.chord.Chord(n.pitches, quarterLength=n.quarterLength)
                    for i in range(len(n.pitches)):
                        if n[i].tie:
                            c[i].tie = m21.tie.Tie(n[i].tie.type)
                    measure_new.append(c)

            measures_new.append(measure_new)
                
    right_hand_new, left_hand_new = m21.stream.Part(), m21.stream.Part()
    right_hand_new.append(measures_right_new)
    left_hand_new.append(measures_left_new)

    s_new = m21.stream.Score()
    s_new.append([right_hand_new, left_hand_new])
    # clefs
    s_new[0][0].insert(0, m21.clef.TrebleClef())
    s_new[1][0].insert(0, m21.clef.BassClef())

    return s_new
    
    
def quantise_score(s):
    if len(list(s.parts)) == 2:
        right_hand, left_hand = tuple(s.parts)
    elif len(list(s.parts)) == 3:
        right_hand, left_hand, _ = tuple(s.parts)

    right_hand, left_hand = right_hand.chordify(), left_hand.chordify()
    measures_right = list(right_hand.getElementsByClass(m21.stream.Measure))
    measures_left = list(left_hand.getElementsByClass(m21.stream.Measure))
    N_bars = len(measures_right)
    
    measures_right_new, measures_left_new = [], []
    for measures_new, measures in [(measures_right_new, measures_right), (measures_left_new, measures_left)]:
        for bar_index in range(N_bars):
            measure_new = m21.stream.Measure()

            k = list(measures[bar_index].getElementsByClass(m21.key.KeySignature))
            if len(k) > 0:
                measure_new.append(m21.key.KeySignature(k[0].sharps))

            t = list(measures[bar_index].getElementsByClass(m21.meter.TimeSignature))
            if len(t) > 0:
                measure_new.append(m21.meter.TimeSignature(t[0].ratioString))

            for ni, n in enumerate(measures[bar_index].notesAndRests):
                duration = round((n.offset + n.quarterLength - measure_new.quarterLength)*8)/8
                if duration == 0: continue

                if type(n) == m21.note.Rest:
                    r = m21.note.Rest(quarterLength=duration)
                    measure_new.append(r)
                elif type(n) == m21.chord.Chord:
                    c = m21.chord.Chord(n.pitches, quarterLength=duration)
                    for i in range(len(n.pitches)):
                        if n[i].tie:
                            c[i].tie = m21.tie.Tie(n[i].tie.type)
                    measure_new.append(c)

            measures_new.append(measure_new)
                
    right_hand_new, left_hand_new = m21.stream.Part(), m21.stream.Part()
    right_hand_new.append(measures_right_new)
    left_hand_new.append(measures_left_new)
    
    s_new = m21.stream.Score()
    s_new.append([right_hand_new, left_hand_new])
    # clefs
    s_new[0][0].insert(0, m21.clef.TrebleClef())
    s_new[1][0].insert(0, m21.clef.BassClef())
    return s_new
            
    
def get_downbeats_and_end_time(midi_data, time_threshold=0.05):
    downbeats = midi_data.get_downbeats()
    end_time = midi_data.get_end_time()
    if downbeats[-1] >= end_time - time_threshold:
        downbeats = downbeats[:-1]
    return downbeats, end_time
    
    
    