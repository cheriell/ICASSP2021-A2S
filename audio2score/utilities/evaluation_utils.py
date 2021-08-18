import os
import sys
from typing import Optional
import numpy as np
import torch
import mir_eval
from jiwer import wer
from subprocess import check_output


class Eval():

    @staticmethod
    def framewise_evaluation_numpy(output: np.ndarray, target: np.ndarray):
        """Framewise F-measure on pianoroll transcription.

            Args:
                output: (dtype: bool) pianoroll prediction.
                target: (dtype: bool) pianoroll ground truth.
            Returns:
                p: (float) precision.
                r: (float) recall.
                f: (float) f-score.
                a: (float) accuracy.
            """
        # get true positive, false positive and false negative
        TP = np.logical_and(output == True, target == True).sum()
        FP = np.logical_and(output == True, target == False).sum()
        FN = np.logical_and(output == False, target == True).sum()

        # get precision, recall, f-score and accuracy
        p = TP / (TP + FP + np.finfo(float).eps)
        r = TP / (TP + FN + np.finfo(float).eps)
        f = 2 * p * r / (p + r + np.finfo(float).eps)
        a = TP / (TP + FP + FN + np.finfo(float).eps)
        
        return p, r, f, a

    @staticmethod
    def framewise_evaluation(output: torch.Tensor, target: torch.Tensor):
        """Framewise f-measure on pianoroll transcription, inputs as torch tensors.

            Args:
                output: (dtype: bool) pianoroll prediction.
                target: (dtype: bool) pianoroll ground truth.
            Returns:
                p: (float) precision.
                r: (float) recall.
                f: (float) f-score.
                a: (float) accuracy.
            """
        
        TP = torch.logical_and(output == True, target == True).sum()
        FP = torch.logical_and(output == True, target == False).sum()
        FN = torch.logical_and(output == False, target == True).sum()
        
        p = TP / (TP + FP + np.finfo(float).eps)
        r = TP / (TP + FN + np.finfo(float).eps)
        f = 2 * p * r / (p + r + np.finfo(float).eps)
        acc = TP / (TP + FP + FN + np.finfo(float).eps)
        
        return p.item(), r.item(), f.item(), acc.item()


    @staticmethod
    def get_notes_intervals(piano_roll: torch.Tensor, hop_time: Optional[float] = 0.01):
        
        piano_roll_padded = torch.Tensor(np.zeros((88, piano_roll.shape[1] + 2), dtype=int))
        piano_roll_padded[:,1:-1] = piano_roll.int()
        
        pr_filter = piano_roll_padded[:,1:] - piano_roll_padded[:,:-1]
        onsets = (pr_filter == 1).nonzero()
        offsets = (pr_filter == -1).nonzero()
        
        notes = onsets[:,0] + 21
        intervals = torch.cat([onsets[:,1].unsqueeze(1), offsets[:,1].unsqueeze(1)], dim=1) * hop_time
        
        return notes.cpu().detach().numpy(), intervals.cpu().detach().numpy()
        
    @staticmethod
    def notewise_evaluation(piano_roll_target: torch.Tensor, 
                            piano_roll_output: torch.Tensor, 
                            hop_time: float):
        notes_target, intervals_target = Eval.get_notes_intervals(piano_roll_target, hop_time)
        if len(notes_target) == 0:
            return None, None
        notes_output, intervals_output = Eval.get_notes_intervals(piano_roll_output, hop_time)
        
        match_on = mir_eval.transcription.match_notes(intervals_target, notes_target,
                                                    intervals_output, notes_output,
                                                    offset_ratio=None, pitch_tolerance=0.25)
        match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target,
                                                    intervals_output, notes_output, 
                                                    offset_ratio=0.2, pitch_tolerance=0.25)
        
        P_n_on = float(len(match_on))/(len(notes_output)+np.finfo(float).eps)
        R_n_on = float(len(match_on))/(len(notes_target)+np.finfo(float).eps)
        F_n_on = 2*P_n_on*R_n_on/(P_n_on+R_n_on+np.finfo(float).eps)

        P_n_onoff = float(len(match_onoff))/(len(notes_output)+np.finfo(float).eps)
        R_n_onoff = float(len(match_onoff))/(len(notes_target)+np.finfo(float).eps)
        F_n_onoff = 2*P_n_onoff*R_n_onoff/(P_n_onoff+R_n_onoff+np.finfo(float).eps)
        
        return (P_n_on, R_n_on, F_n_on), (P_n_onoff, R_n_onoff, F_n_onoff)

    @staticmethod
    def wer_evaluation(target: str, output: str):
        return wer(target, output)

    @staticmethod
    def mv2h_evaluation(target_midi_file: str, output_midi_file: str, MV2H_path: str, timeout: Optional[float] = 10.):
        try:
            output = check_output(['sh', os.path.join('audio2score', 'utilities', 'evaluate_midi_mv2h.sh'), 
                                                    target_midi_file, output_midi_file, MV2H_path], 
                                                    timeout=timeout)
        except ValueError as e:
            print('Failed to evaluate pair: \ntarget midi: {}\noutput midi: {}'.format(target_midi_file,
                                                                            output_midi_file))

        # extract result from output
        result_list = output.decode('utf-8').splitlines()[-6:]
        result = dict([tuple(item.split(': ')) for item in result_list])
        for key, value in result.items():
            result[key] = float(value)
        
        return result