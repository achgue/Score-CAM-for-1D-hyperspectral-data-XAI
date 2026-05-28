import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM1D(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        # different from original since we are using a 1D implementation
        # l is the number of channels in the spectrum
        # batch, lenght, width, height
        b, l, w, h = input.size()


        # takes a one dimesional tensor
        # dato che é un unica valutazione ci basta un tensore unidimensionale
        score_saliency_map = torch.zeros((1, l, 1, 1))
        # predication on raw input
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        

        logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
          logit = logit.cuda()


        #self.model_arch.zero_grad()
        #score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        
        # k è il numero di filtri
        b, k, u = activations.size()


        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
          for i in range(k):
              #print(activations.shape)
              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :], 1)
              saliency_map = F.interpolate(saliency_map, size=l, mode='linear', align_corners=False)

              saliency_map = saliency_map.permute(0, 2, 1).unsqueeze(-1)
              if saliency_map.max() == saliency_map.min():
                continue
              
              # normalize to 0-1
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              # calcola miglioramento se effettuo unaltra predizione sul sample mascherato
              # la saliency normalizzata diventa la nostra machera
              output = self.model_arch(input * norm_saliency_map)
              # devo verificare che prendendo la 0 prendo quella 
              class_probabilities = F.softmax(output, dim=1) # result from the network
              score = class_probabilities[0][predicted_class] # value for the predicted class

              score_saliency_map +=  score * saliency_map
        
        
        score_saliency_map = F.relu(score_saliency_map)

        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)