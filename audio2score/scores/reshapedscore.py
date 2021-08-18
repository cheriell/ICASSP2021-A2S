import sys
import music21 as m21
import numpy as np

from audio2score.scores.symbols import *



class ScoreReshaped(object):
    """Reshaped Score Representation 
    
        (only one measure for one part). Ties are added to the stop and continue notes
        Args:
            names: pitch names in numpy array
            octaves: pitch octaves in numpy array
            ties: ties in numpy array
            duration: note durations in numpy array
        """

    def __init__(self, names: np.ndarray, octaves: np.ndarray, ties: np.ndarray, duration: np.ndarray):

        if names.shape[0] != 5 or octaves.shape[0] != 5 or ties.shape[0] != 5:
            print(names.shape, octaves.shape, ties.shape)
            sys.exit('Dimension Error! please check input dimension!')
        elif names.shape[1] != duration.shape[0] or octaves.shape[1] != duration.shape[0] or ties.shape[1] != duration.shape[0]:
            print(names.shape, octaves.shape, ties.shape, duration.shape)
            sys.exit('Length Error! length not equal, please check input length!')
        
        self.length   = duration.shape[0]
        self.names    = names      # [5, length]
        self.octaves  = octaves    # [5, length]
        self.ties     = ties       # [5, length]
        self.duration = duration   # [length]
        self.grammarise()
        
    @classmethod
    def from_m21(cls, measure: m21.stream.Measure):
        def split_duration(d):
            if d.quarterLength in quarterLength2duration:
                return [d]
            ds = []
            for l in [4, 2, 1, 0.5, 0.25, 0.125]:
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
                if n.tie and n.tie.type in {'continue', 'stop'}:
                    ties.append('~')
                else:
                    ties.append('_')
                ties_extra.append('~')
            elif type(n) == m21.chord.Chord:
                for pi, p in enumerate(n.pitches):
                    names.append(Name2name[p.name])
                    octaves.append(Octave2octave[p.octave])
                    if n[pi].tie and n[pi].tie.type in {'continue', 'stop'}:
                        ties.append('~')
                    else:
                        ties.append('_')
                    ties_extra.append('~')
            elif type(n) == m21.note.Rest:
                names.append('r')
                octaves.append('_')
                ties.append('_')
                ties_extra.append('_')
            names = names[:5] + ['_'] * max(0, 5-len(names))
            octaves = octaves[:5] + ['_'] * max(0, 5-len(octaves))
            ties = ties[:5] + ['_'] * max(0, 5-len(ties))
            ties_extra = ties_extra[:5] + ['_'] * max(0, 5-len(ties_extra))
            return names, octaves, ties, ties_extra
        
        # get ScoreReshaped from m21.stream.Measure
        names, octaves, ties, duration = [], [], [], []
        for n in measure.flat.notesAndRests:
            duration_cur = split_duration(n.duration)
            names_cur, octaves_cur, ties_cur, ties_extra_cur = get_names_octaves_ties(n)
            
            for i in range(len(duration_cur)):
                names.append(names_cur)
                octaves.append(octaves_cur)
                ties.append(ties_cur if i == 0 else ties_extra_cur)
                duration.append(quarterLength2duration[duration_cur[i].quarterLength])
        
        names = np.transpose(np.array(names), axes=(1, 0))
        octaves = np.transpose(np.array(octaves), axes=(1, 0))
        ties = np.transpose(np.array(ties), axes=(1, 0))
        duration = np.array(duration)
        return cls(names, octaves, ties, duration)
    
    @classmethod
    def from_index_matrix(cls, index_matrix: np.ndarray):
        if index_matrix.shape[1] == 0:
            names = np.array([['r'], ['_'], ['_'], ['_'], ['_']])
            octaves = np.array([['_'], ['_'], ['_'], ['_'], ['_']])
            ties = np.array([['_'], ['_'], ['_'], ['_'], ['_']])
            duration = np.array(['4'])
        else:
            names = np.vectorize(index2name.get)(index_matrix[:5])
            octaves = np.vectorize(index2octave.get)(index_matrix[5:10])
            ties = np.vectorize(index2tie.get)(index_matrix[10:15])
            duration = np.vectorize(index2duration.get)(index_matrix[15])
        return cls(names, octaves, ties, duration)
    
    def grammarise(self):
        # remove all reserved symbols
        for n in [self.names, self.octaves, self.ties, self.duration]:
            n[n == '<SOS>'] = '_'
            n[n == '<EOS>'] = '_'
            n[n == '<PAD>'] = '_'
            
        # remove columns with no duration
        valid_durations = self.duration[:] != '_'
        self.names = self.names[:,valid_durations]
        self.octaves = self.octaves[:,valid_durations]
        self.ties = self.ties[:,valid_durations]
        self.duration = self.duration[valid_durations]
        self.length = self.duration.shape[0]

        for t in range(self.length):
            # is rest
            if np.logical_or(self.names[:,t] == 'r', self.names[:,t] == '_').sum() == 5:
                self.names[0,t] = 'r'
                self.names[1:,t] = '_'
                self.octaves[:,t] = '_'
                self.ties[:,t] = '_'
            else:  # is note or chord
                for poly in range(5):  # filter out rest symbols and fill empty octaves when there is a note
                    if self.names[poly,t] == 'r':
                        self.names[poly,t] = '_'  # set to empty
                    elif self.names[poly,t] != '_' and self.octaves[poly,t] == '_':
                        self.octaves[poly,t] = '-'  # set to default
                # remove duplicate pitches
                pitches = []
                pitches_and_ties = []
                for poly in range(5):
                    if self.names[poly,t] != '_':
                        pitch = (self.names[poly,t], self.octaves[poly,t])
                        pitch_and_tie = (self.names[poly,t], self.octaves[poly,t], self.ties[poly,t])
                        if pitch not in pitches:
                            pitches_and_ties.append(tuple(pitch_and_tie))
                        pitches.append(tuple(pitches))
                self.names[:,t] = '_'
                self.octaves[:,t] = '_'
                for poly, (name, octave, tie) in enumerate(pitches_and_ties):
                    self.names[poly,t] = name
                    self.octaves[poly,t] = octave
                    self.ties[poly,t] = tie
                # grammarise ties
                for poly in range(5):
                    if self.names[poly,t] in ['_', 'r']:
                        self.ties[poly,t] = '_'  # set to empty
        
    def to_matrix(self):
        return np.concatenate((self.names, self.octaves, self.ties, np.expand_dims(self.duration, axis=0)), axis=0)
        
    def to_m21(self):
        measure = m21.stream.Measure()
        for t in range(self.length):
            quarter_length = duration2quarterLength[self.duration[t]]
            if self.names[0,t] == 'r':
                rest = m21.note.Rest(quarterLength=quarter_length)
                measure.append(rest)
            elif self.names[1,t] == '_':
                pitch = name2Name[self.names[0,t]]+str(octave2Octave[self.octaves[0,t]])
                note = m21.note.Note(pitch, quarterLength=quarter_length)
                measure.append(note)
            else:
                pitches = [name2Name[self.names[poly,t]]+str(octave2Octave[self.octaves[poly,t]]) for poly in range(5) if self.names[poly,t] != '_']
                chord = m21.chord.Chord(pitches, quarterLength=quarter_length)
                measure.append(chord)
        # add ties backwardly
        next_tied_pitches = set()
        for t in range(self.length)[::-1]:
            n = measure[t]
            notes = [n] if type(n) == m21.note.Note else [n[i] for i in range(len(n))] if type(n) == m21.chord.Chord else []
            new_tied_pitches = set([notes[i].pitch.midi for i in range(len(notes)) if self.ties[i,t] == '~'])
            for note in notes:
                if note.pitch.midi in next_tied_pitches:   # to be continued
                    if note.pitch.midi in new_tied_pitches:
                        note.tie = m21.tie.Tie('continue')
                    else:
                        note.tie = m21.tie.Tie('start')
                        next_tied_pitches.remove(note.pitch.midi)
                else: # not to be continued:
                    if note.pitch.midi in new_tied_pitches:
                        note.tie = m21.tie.Tie('stop')
                    # else do nothing
            next_tied_pitches.update(new_tied_pitches)
        return measure
    
    def to_index_matrix(self):
        index_names = np.vectorize(name2index.get)(self.names)
        index_octaves = np.vectorize(octave2index.get)(self.octaves)
        index_ties = np.vectorize(tie2index.get)(self.ties)
        index_duration = np.vectorize(duration2index.get)(self.duration)
        return np.concatenate((index_names, index_octaves, index_ties, np.expand_dims(index_duration, axis=0)), axis=0)
        
