import pandas as pd
import torch
from torch import nn


class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False,
                 prompt_key=False, pool_size=None, top_k=None,batchwise_prompt=False, prompt_key_init='uniform',num_classes=None,ways=None,num_tasks=None):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_classes=num_classes
        self.ways=ways
        self.num_tasks=num_tasks
        self.start = 0
        if self.num_classes==200:
            self.num_base=100
        elif self.num_classes==100:
            self.num_base=60

        if self.prompt_pool:
            self.prompt_list=[]
            for i in range(num_tasks):
                prompt_pool_shape = (pool_size, length, embed_dim)
                if prompt_init == 'zero':
                    prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    self.register_parameter('prompt_%d' % i, prompt)
                    self.prompt_list.append(prompt)
                elif prompt_init == 'uniform':
                    prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    self.register_parameter('prompt_%d' % i, prompt)
                    nn.init.uniform_(prompt, -1, 1)
                    self.prompt_list.append(prompt)

        if self.prompt_key:
            self.prompt_key_list = []
            for i in range(num_tasks):
                key_shape = (pool_size, embed_dim)
                if prompt_key_init == 'zero':
                    prompt_key = nn.Parameter(torch.zeros(key_shape))
                    self.register_parameter('prompt_key_%d' % i, prompt_key)
                    self.prompt_key_list.append(prompt_key)
                elif prompt_key_init == 'uniform':
                    prompt_key = nn.Parameter(torch.randn(key_shape))
                    self.register_parameter('prompt_key_%d' % i, prompt_key)
                    nn.init.uniform_(prompt_key, -1, 1)
                    self.prompt_key_list.append(prompt_key)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None,prompts_matrix=None,is_training=None,task_id=-1):
        self.is_training=is_training
        self.task_id=task_id
        # print(task_id,is_training)
        # assert self.is_training is not None and self.is_training is not None
        if self.is_training:
            prompt = self.prompt_list[self.task_id]
            prompt_key = self.prompt_key_list[self.task_id]
            
            self.start = 0
        else:
            prompt_list = []
            prompt_key_list = []
            for i in range(self.task_id + 1):
                prompt_list.append(self.prompt_list[i])
                prompt_key_list.append(self.prompt_key_list[i])
            prompt = torch.cat(prompt_list, dim=0)
            
        out = dict()

        assert self.prompt_pool==True
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0]  # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_key_norm = self.l2_normalize(prompt_key, dim=1)  # Pool_size*ways, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # B, C

            # if self.training:
            #     similarity = torch.matmul(x_embed_norm, prompt_key_norm.t())  # B, Pool_size
            #     similarity = torch.mul(similarity, prompts_matrix)
            #     similarity[similarity == 0] = float("-inf")
            # else:

            # import time
            # start_time = time.time()
            similarity = torch.matmul(x_embed_norm, prompt_key_norm.t())

            assert prompt_mask is None, "prompt_mask must be None!"
            if prompt_mask is None:
                topk_similarity, idx = torch.topk(similarity, k=self.top_k, dim=1)

                # if self.batchwise_prompt:
                #     prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                #     # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                #     # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                #     # Unless dimension is specified, this will be flattend if it is not already 1D.
                #     if prompt_id.shape[0] < self.pool_size:
                #         prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],),
                #                                                      torch.min(idx.flatten()),
                #                                                      device=prompt_id.device)])
                #         id_counts = torch.cat(
                #             [id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                #     _, major_idx = torch.topk(id_counts, k=self.top_k)  # top_k
                #     major_prompt_id = prompt_id[major_idx]  # top_k
                #     # expand to batch
                #     idx = major_prompt_id.expand(x_embed.shape[0], -1)  # B, top_k

            else:
                idx = prompt_mask  # B, top_k

            batched_prompt_raw = prompt[idx]  # B, top_k, length, C
            
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)  # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_key_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm.reshape(batch_size, top_k * c)
            out['selected_prompt']=batched_prompt.reshape(batch_size, top_k * length * c)
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar

            out['reduce_sim'] = reduce_sim
            out['topk_similarity'] = topk_similarity
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        # out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        out['prompted_embedding'] = torch.cat([x_embed], dim=1)

        return out
