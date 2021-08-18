
###################################################
## all symbols and index dicts
###################################################

NON, PAD, SOS, EOS = 0, 1, 2, 3
reserved_symbols = ['_', '<PAD>', '<SOS>', '<EOS>']

names_set = ['c', 'd', 'e', 'f', 'g', 'a', 'b', 'cis', 'dis', 'eis', 'fis', 'gis', 'ais', 'bis', 'ces', 'des', 'ees', 'fes', 'ges', 'aes', 'bes', 'cisis', 'fisis', 'beses']
name2Name = {'c':'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'a': 'A', 'b': 'B', 'cis': 'C#', 'dis': 'D#', 'eis': 'E#', 'fis': 'F#', 'gis': 'G#', 'ais': 'A#', 'bis': 'B#', 'ces':'C-', 'des': 'D-', 'ees': 'E-', 'fes': 'F-', 'ges': 'G-', 'aes': 'A-', 'bes': 'B-', 'cisis': 'C##', 'fisis': 'F##', 'beses': 'B--'}
Name2name = dict([(Name, name) for name, Name in name2Name.items()])
name2index = dict([(name, index) for index, name in enumerate(reserved_symbols + names_set + ['r'])])
index2name = dict([(index, name) for index, name in enumerate(reserved_symbols + names_set + ['r'])])
vocab_size_name = len(name2index)

octaves_set = [',,,', ',,', ',', '-', "'", "''", "'''", "''''", "'''''"]
octave2Octave = {',,,': 0, ',,': 1, ',': 2, '-': 3, "'": 4, "''": 5, "'''": 6, "''''": 7, "'''''": 8}
Octave2octave = dict([(Octave, octave) for octave, Octave in octave2Octave.items()])
octave2index = dict([(octave, index) for index, octave in enumerate(reserved_symbols + octaves_set)])
index2octave = dict([(index, octave) for index, octave in enumerate(reserved_symbols + octaves_set)])
vocab_size_octave = len(octave2index)

tie2index = dict([(tie, index) for index, tie in enumerate(reserved_symbols + ['~'])])
index2tie = dict([(index, tie) for index, tie in enumerate(reserved_symbols + ['~'])])
vocab_size_tie = len(tie2index)

durations_set = ['1', '2', '4', '8', '16', '32', '1.', '2.', '4.', '8.', '16.', '32.', '\\breve']
duration2quarterLength = {'1': 4, '2': 2, '4': 1, '8': 0.5, '16': 0.25, '32': 0.125, '1.': 6, '2.': 3, '4.': 1.5, '8.': 0.75, '16.': 0.375, '32.': 0.1875, '\\breve': 8}
quarterLength2duration = dict([(quarterLength, duration) for duration, quarterLength in duration2quarterLength.items()])
duration2index = dict([(duration, index) for index, duration in enumerate(reserved_symbols + durations_set)])
index2duration = dict([(index, duration) for index, duration in enumerate(reserved_symbols + durations_set)])
vocab_size_duration = len(duration2index)

words_set = durations_set + ['~'] + names_set + ['r'] + octaves_set + ['<', '>']
word2index = dict([(word, index) for index, word in enumerate(reserved_symbols + words_set)])
index2word = dict([(index, word) for index, word in enumerate(reserved_symbols + words_set)])
vocab_size = len(word2index)
