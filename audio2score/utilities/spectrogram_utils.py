import librosa
import numpy as np


class SpectrogramUtil():

    FMIN = librosa.note_to_hz('A0')
    FMAX = 44100 / 2

    @staticmethod
    def HCQT_from_file(audio_file, 
                        bins_per_octave=60, 
                        n_octaves=6, 
                        n_harms=6):

        harmonics_list = [0.5, 1, 2, 3, 4, 5]
        
        y, fs = librosa.load(audio_file, sr=25600)

        cqt_list = []
        shapes = []
        for h in harmonics_list[:n_harms]:
            cqt = librosa.cqt(y, sr=fs, 
                                hop_length=256, 
                                fmin=SpectrogramUtil.FMIN*float(h), 
                                n_bins=bins_per_octave*n_octaves, 
                                bins_per_octave=bins_per_octave)
            cqt_list.append(cqt)
            shapes.append(cqt.shape)

        shapes_equal = [s == shapes[0] for s in shapes]
        if not all(shapes_equal):
            min_time = np.min([s[1] for s in shapes])
            new_cqt_list = []
            for i in range(len(cqt_list)):
                new_cqt_list.append(cqt_list[i][:, :min_time])
            cqt_list = new_cqt_list

        log_hcqt = ((1./80.) * librosa.amplitude_to_db(np.abs(np.array(cqt_list)), ref=np.max)) + 1.

        return log_hcqt


    @staticmethod
    def VQT_from_file(audio_file, 
                        bins_per_octave=60, 
                        n_octaves=8, 
                        gamma=20):
        y, fs = librosa.load(audio_file, sr=25600)
        
        vqt = librosa.vqt(y, sr=fs, 
                            hop_length=256, 
                            fmin=SpectrogramUtil.FMIN, 
                            n_bins=bins_per_octave*n_octaves, 
                            bins_per_octave=bins_per_octave, 
                            gamma=gamma)
        
        log_vqt = ((1./80.) * librosa.amplitude_to_db(np.abs(np.array(vqt)), ref=np.max)) + 1.
        
        return log_vqt


    @staticmethod
    def CQT_from_file(audio_file, 
                    bins_per_octave=12, 
                    n_octaves=7):
        y, fs = librosa.load(audio_file, sr=25600)
        fmin = librosa.note_to_hz('C1')
        
        cqt = librosa.cqt(y, sr=fs, 
                            hop_length=256, 
                            fmin=fmin, 
                            n_bins=bins_per_octave*n_octaves, 
                            bins_per_octave=bins_per_octave)
        
        log_cqt = ((1./80.) * librosa.amplitude_to_db(np.abs(np.array(cqt)), ref=np.max)) + 1.
        
        return log_cqt


    @staticmethod
    def STFT_from_file(audio_file, win_length=1024):
        y, fs = librosa.load(audio_file, sr=44100)
        stft_fmax=11025
        stft_frequency_filter = librosa.fft_frequencies(sr=fs, n_fft=2048) <= stft_fmax
        
        stft = librosa.stft(y, hop_length=441, win_length=win_length)
        stft = stft[stft_frequency_filter]
        
        log_stft = ((1./80.) * librosa.amplitude_to_db(np.abs(np.array(stft)), ref=np.max)) + 1.
        
        return log_stft


    @staticmethod
    def melspectrogram_from_file(audio_file, 
                                win_length=2048, 
                                n_mels=256):
        y, fs = librosa.load(audio_file, sr=44100)

        melspec = librosa.feature.melspectrogram(y, sr=fs, 
                                        win_length=win_length, 
                                        hop_length=441, 
                                        n_mels=n_mels, 
                                        fmin=SpectrogramUtil.FMIN, 
                                        fmax=SpectrogramUtil.FMAX)

        log_mel = ((1./80.) * librosa.amplitude_to_db(np.abs(np.array(melspec)), ref=np.max)) + 1.

        return log_mel