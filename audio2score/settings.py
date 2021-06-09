

class Constants:
    hop_time = 0.01
    segment_length = 4  # transcribe 4s segment at a time
    input_length = int(segment_length * (1 / hop_time))  # model input length corresponds to segment length
    velocity_threshold = 30

    pianos = ['Gentleman', 'Giant', 'Grandeur', 'Maverick']

class TrainingParam:
    learning_rate = 0.001
    gamma = 0.9
    batch_size = 8

class SpectrogramSetting(object):
    type = 'VQT'  # select from ['STFT', 'Mel', 'CQT', 'HCQT', 'VQT']
    win_length = 2048
    n_mels = 256
    bins_per_octave = 60
    n_octaves = 8
    n_harms = 6
    gamma = 20
    freq_bins = 480
    channels = 1

    def to_string(self):
        if self.type == 'STFT':
            return '-'.join([self.type,
                        f'win_length={self.win_length}'])
        elif self.type == 'Mel':
            return '-'.join([self.type,
                        f'win_length={self.win_length}',
                        f'n_mels={self.n_mels}'])
        elif self.type == 'CQT':
            return '-'.join([self.type,
                        f'bins_per_octave={self.bins_per_octave}',
                        f'n_octaves={self.n_octaves}'])
        elif self.type == 'HCQT':
            return '-'.join([self.type,
                        f'bins_per_octave={self.bins_per_octave}',
                        f'n_octaves={self.n_octaves}',
                        f'n_harms={self.n_harms}'])
        elif self.type == 'VQT':
            return '-'.join([self.type,
                        f'bins_per_octave={self.bins_per_octave}',
                        f'n_octaves={self.n_octaves}',
                        f'gamma={self.gamma}'])
