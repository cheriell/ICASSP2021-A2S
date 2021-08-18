import sys
from typing import Optional
import music21 as m21


class ScoreM21(object):
    """ScoreM21 for only one bar

        Args:
            key_signature: key signature
            time_signature: time signature
            right: right hand score
            left: left hand score
            prev_tied_pitches: previous tied pitches
            next_tied_pitches: pitches to be tied in the next bars
        """

    def __init__(self, key_signature: m21.key.KeySignature,
                        time_signature: m21.meter.TimeSignature,
                        right: m21.stream.Measure,
                        left: m21.stream.Measure,
                        prev_tied_pitches: Optional[dict] = None,
                        next_tied_pitches: Optional[dict] = None):
            
        if not prev_tied_pitches:
            prev_tied_pitches = self.init_prev_tied_pitches()
        if not next_tied_pitches:
            next_tied_pitches = self.init_next_tied_pitches()

        self.key_signature = key_signature  # m21.key.KeySignature
        self.time_signature = time_signature   # m21.meter.TimeSignature
        self.right = right  # m21.stream.Measure
        self.left = left   # m21.stream.Measure
        self.prev_tied_pitches, self.next_tied_pitches = self.grammarise(prev_tied_pitches, next_tied_pitches)
        
    @classmethod
    def init_prev_tied_pitches(cls):
        prev_tied_pitches = dict(right=set(), left=set())
        return prev_tied_pitches
    
    @classmethod
    def init_next_tied_pitches(cls):
        next_tied_pitches = dict(right=set(), left=set())
        return next_tied_pitches
        
    @classmethod
    def from_m21(cls, filename: str):
        s = m21.converter.parse(filename)
        if len(s.parts) != 2:
            sys.exit('Invalid file, only two parts are allowed!')
        right_part, left_part = tuple(s.parts)
        
        right_measures = list(right_part.getElementsByClass(m21.stream.Measure))
        left_measures = list(left_part.getElementsByClass(m21.stream.Measure))
        if len(right_measures) != 1 or len(left_measures) != 1:
            sys.exit('Invalid file, only one bar allowed!')
        right = right_measures[0]
        left = left_measures[0]
        
        k = list(right.getElementsByClass(m21.key.KeySignature))
        t = list(right.getElementsByClass(m21.meter.TimeSignature))
        if len(k) == 0 or len(t) == 0:
            sys.exit('Invalid file, no key and time signatures!')
        key_signature = m21.key.KeySignature(k[0].sharps)
        time_signature = m21.meter.TimeSignature(t[0].ratioString)
        
        return cls(key_signature, time_signature, right, left)
            
    def grammarise(self, prev_tied_pitches: dict, next_tied_pitches: dict):
        def grammarise_measure(measure, prevs, nexts):
            # forward check
            for n in measure.notesAndRests:
                notes = [n] if type(n) == m21.note.Note else [n[i] for i in range(len(n))] if type(n) == m21.chord.Chord else []
                new_tied_pitches = set([note.pitch.midi for note in notes if note.tie and note.tie.type in {'start', 'continue'}])
                for note in notes:
                    if note.pitch.midi in prevs:  # is continued
                        if note.pitch.midi in new_tied_pitches:
                            note.tie = m21.tie.Tie('continue')
                        else:
                            note.tie = m21.tie.Tie('stop')
                            prevs.remove(note.pitch.midi)
                prevs.update(new_tied_pitches)
            # backward check
            for n in list(measure.notesAndRests)[::-1]:
                notes = [n] if type(n) == m21.note.Note else [n[i] for i in range(len(n))] if type(n) == m21.chord.Chord else []
                new_tied_pitches = set([note.pitch.midi for note in notes if note.tie and note.tie.type in {'stop', 'continue'}])
                for note in notes:
                    if note.pitch.midi in nexts:  # is to be continued
                        if note.pitch.midi in new_tied_pitches:
                            note.tie = m21.tie.Tie('continue')
                        else:
                            note.tie = m21.tie.Tie('start')
                            nexts.remove(note.pitch.midi)
                nexts.update(new_tied_pitches)
            return prevs, nexts
        prev_tied_pitches['right'], next_tied_pitches['right'] = grammarise_measure(self.right, prev_tied_pitches['right'], next_tied_pitches['right'])
        prev_tied_pitches['left'], next_tied_pitches['left'] = grammarise_measure(self.left, prev_tied_pitches['left'], next_tied_pitches['left'])
        return prev_tied_pitches, next_tied_pitches
        
    def save_to_xml(self, filename: str):
        if filename[-4:] != '.xml':
            sys.exit('Input Error! filename should end with .xml')
        s = m21.stream.Score()
        s.append([m21.stream.Part(1), m21.stream.Part(2)])
        s[0].append(self.right)
        s[0][0].insert(0, m21.meter.TimeSignature(self.time_signature.ratioString))
        s[0][0].insert(0, m21.key.KeySignature(self.key_signature.sharps))
        s[1].append(self.left)
        s[1][0].insert(0, m21.meter.TimeSignature(self.time_signature.ratioString))
        s[1][0].insert(0, m21.key.KeySignature(self.key_signature.sharps))
        s.write('xml', filename)
        