import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_size)
        self.item_embedding = nn.Embedding(num_items, hidden_size)

    def embeddings(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        return user_emb, item_emb

    def forward(self, input):
        user_emb, item_emb = self.embeddings(user_ids, item_ids)

        pred_rating = torch.sum(torch.mul(user_emb, item_emb), 1)
        return pred_rating

    def process_one_batch(self, users, items, ratings):
        pos_ratings = self.forward(users, items)
        if self.pointwise:
            loss = self.loss_func(pos_ratings, ratings)
        else:
            neg_ratings = self.forward(users, ratings)
            loss = -F.sigmoid(pos_ratings - neg_ratings).log().mean()

        return loss

    def predict_batch_users(self, user_ids):
        user_latent = self.user_embedding(user_ids)
        all_item_latent = self.item_embedding.weight.data
        return user_latent @ all_item_latent.T

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for b in range(num_batches):
                if (b + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[b * test_batch_size:]
                else:
                    batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]

                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users).to(self.device)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()

        pred_matrix[eval_pos.nonzero()] = float('-inf')
        return pred_matrix
