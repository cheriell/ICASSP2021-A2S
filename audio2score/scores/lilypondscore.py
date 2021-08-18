import sys
import music21 as m21
import numpy as np

from audio2score.scores.symbols import *


class ScoreLilyPond(object):
    """Lilypond Score Representation 

        (only one measure for one part) Ties are added to the start and continue notes
        Args:
            names: list of pitch names
            octaves: list of pitch octaves
            ties: list of ties
            duration: lisit of note durations
        """

    def __init__(self, names: list, octaves: list,
                        ties: list, duration: list):

        if len(names) != len(duration) or len(octaves) != len(duration) or len(ties) != len(duration):
            sys.exit('Length Error! length not equal, please check input length!')
            
        self.length = len(duration)
        for t in range(self.length):
            if type(names[t]) != list or type(octaves[t]) != list or type(ties[t]) != list:
                sys.exit('Type Error!')
            if len(names) != len(ties) or len(octaves) != len(ties):
                sys.exit('Poly level Error!')
                
        self.names = names         # [length, poly[t]]
        self.octaves = octaves     # [length, poly[t]]
        self.ties = ties           # [length, poly[t]]
        self.duration = duration   # [length]
        self.grammarise()
        
    @classmethod
    def from_m21(cls, measure: m21.stream.Measure):
        def split_duration(d):
            if d.quarterLength in quarterLength2duration:
                return [d]
            ds = []
            for l in [4, 2, 1, 0.5]:
                if d.quarterLength > l:
                    ds.append(m21.duration.Duration(l))
                    ds += split_duration(m21.duration.Duration(d.quarterLength-l))
                    break
            return ds
        
        def get_names_octaves_ties(n):
            names, octaves, ties, ties_extra = [], [], [], []
            if type(n) == m21.note.Note:
                names.append(Name2name[n.pitch.name])
                octaves.append(Octave2octave[n.pitch.octave])
                if n.tie and n.tie.type in {'continue', 'start'}:
                    ties.append('~')
                else:
                    ties.append('_')
                ties_extra.append('~')
            elif type(n) == m21.chord.Chord:
                for pi, p in enumerate(n.pitches):
                    names.append(Name2name[p.name])
                    octaves.append(Octave2octave[p.octave])
                    if n[pi].tie and n[pi].tie.type in {'continue', 'start'}:
                        ties.append('~')
                    else:
                        ties.append('_')
                    ties_extra.append('~')
            elif type(n) == m21.note.Rest:
                names.append('r')
                octaves.append('_')
                ties.append('_')
                ties_extra.append('_')
            names = names[:5]
            octaves = octaves[:5]
            ties = ties[:5]
            ties_extra = ties_extra[:5]
            return names, octaves, ties, ties_extra
        
        # get ScoreLilyPond from m21.stream.Measure
        names, octaves, ties, duration = [], [], [], []
        for n in measure.flat.notesAndRests:
            duration_cur = split_duration(n.duration)
            names_cur, octaves_cur, ties_cur, ties_extra_cur = get_names_octaves_ties(n)
            
            for i in range(len(duration_cur)):
                names.append(names_cur)
                octaves.append(octaves_cur)
                ties.append(ties_cur if i == len(duration_cur)-1 else ties_extra_cur)
                duration.append(quarterLength2duration[duration_cur[i].quarterLength])
                
        return cls(names, octaves, ties, duration)
    
    @classmethod
    def from_index_matrix(cls, index_matrix: np.ndarray):
        matrix = np.vectorize(index2word.get)(index_matrix)
        symbols = matrix[0]
        names, octaves, ties, duration = [], [], [], []
        names_cur, octaves_cur, ties_cur = [], [], []
        for si, s in enumerate(symbols):
            if s in names_set:
                names_cur.append(s)
                for i in range(1, 3):
                    if si+i == len(symbols): break
                    if symbols[si+i] in octaves_set:
                        octaves_cur.append(symbols[si+i])
                    elif symbols[si+i] == '~':
                        ties_cur.append(symbols[si+i])
                octaves_cur = octaves_cur[:len(names_cur)]
                ties_cur = ties_cur[:len(names_cur)]
                octaves_cur = octaves_cur + ['_'] * (len(names_cur)-len(octaves_cur))
                ties_cur = ties_cur + ['_'] * (len(names_cur)-len(ties_cur))
            if s in durations_set:
                if len(names_cur) == 0:
                    names_cur, octaves_cur, ties_cur = ['r'], ['_'], ['_']
                names.append(names_cur)
                octaves.append(octaves_cur)
                ties.append(ties_cur)
                duration.append(s)
                names_cur, octaves_cur, ties_cur = [], [], []
        if len(duration) == 0:
            names, octaves, ties, duration = [['r']], [['_']], [['_']], ['4']
        return cls(names, octaves, ties, duration)
    
    
    def grammarise(self):
        for t in range(self.length):
            # is rest:
            if np.array([name in names_set for name in self.names[t]]).sum() == 0:
                self.names[t] = ['r']
                self.octaves[t] = ['_']
                self.ties[t] = ['_']
            else:  # is note or chord
                # filter out rest symbols and fiil empty octaves when there is a note
                poly_level_raw = len(self.names[t])
                for poly in range(poly_level_raw):
                    if self.names[t][poly] in names_set and self.octaves[t][poly] not in octaves_set:
                        self.octaves[t][poly] = '-'  # set to default
                self.octaves[t] = [self.octaves[t][poly] for poly in range(poly_level_raw) if self.names[t][poly] in names_set]
                self.ties[t] = [self.ties[t][poly] for poly in range(poly_level_raw) if self.names[t][poly] in names_set]
                self.names[t] = [self.names[t][poly] for poly in range(poly_level_raw) if self.names[t][poly] in names_set]
                # remove duplicate pitches
                pitches = []
                pitches_and_ties = []
                for poly in range(len(self.names[t])):
                    if self.names[t][poly] in names_set:
                        pitch = tuple([self.names[t][poly], self.octaves[t][poly]])
                        if pitch not in pitches:
                            pitches_and_ties.append(tuple([self.names[t][poly], self.octaves[t][poly], self.ties[t][poly]]))
                        pitches.append(pitch)
                self.names[t] = []
                self.octaves[t] = []
                self.ties[t] = []
                for name, octave, tie in pitches_and_ties:
                    self.names[t].append(name)
                    self.octaves[t].append(octave)
                    self.ties[t].append(tie)
                    
    def to_matrix(self):
        # return shape: [1, lilypond tokens length]
        words = []
        for t in range(self.length):
            if self.names[t][0] == 'r':
                words += ['r', self.duration[t]]
            elif len(self.names[t]) == 1:
                words.append(self.names[t][0])
                if self.octaves[t][0] != '-':
                    words.append(self.octaves[t][0])
                if self.ties[t][0] == '~':
                    words.append('~')
                words.append(self.duration[t])
            else:
                words.append('<')
                for poly in range(len(self.names[t])):
                    words.append(self.names[t][poly])
                    if self.octaves[t][poly] != '-':
                        words.append(self.octaves[t][poly])
                    if self.ties[t][poly] == '~':
                        words.append('~')
                words.append('>')
                words.append(self.duration[t])
        return np.expand_dims(np.array(words), axis=0)
    
    def to_string(self):
        matrix = self.to_matrix()
        return ' '.join(list(matrix[0]))
    
    def to_m21(self):
        measure = m21.stream.Measure()
        for t in range(self.length):
            quarter_length = duration2quarterLength[self.duration[t]]
            if self.names[t][0] == 'r':
                rest = m21.note.Rest(quarterLength=quarter_length)
                measure.append(rest)
            elif len(self.names[t]) == 1:
                pitch = name2Name[self.names[t][0]]+str(octave2Octave[self.octaves[t][0]])
                note = m21.note.Note(pitch, quarterLength=quarter_length)
                measure.append(note)
            else:
                pitches = [name2Name[self.names[t][poly]]+str(octave2Octave[self.octaves[t][poly]]) for poly in range(len(self.names[t]))]
                chord = m21.chord.Chord(pitches, quarterLength=quarter_length)
                measure.append(chord)
        # add ties forwardly
        prev_tied_pitches = set()
        for t in range(self.length):
            n = measure[t]
            notes = [n] if type(n) == m21.note.Note else [n[i] for i in range(len(n))] if type(n) == m21.chord.Chord else []
            new_tied_pitches = set([notes[i].pitch.midi for i in range(len(notes)) if self.ties[t][i] == '~'])
            for note in notes:
                if note.pitch.midi in prev_tied_pitches:   # is continued
                    if note.pitch.midi in new_tied_pitches:
                        note.tie = m21.tie.Tie('continue')
                    else:
                        note.tie = m21.tie.Tie('stop')
                        prev_tied_pitches.remove(note.pitch.midi)
                else:  #  is not continued
                    if note.pitch.midi in new_tied_pitches:
                        note.tie = m21.tie.Tie('start')
                    # else do nothing
            prev_tied_pitches.update(new_tied_pitches)
        return measure
    
    def to_index_matrix(self):
        return np.vectorize(word2index.get)(self.to_matrix())
        