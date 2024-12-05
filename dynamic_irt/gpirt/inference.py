import torch
from amortized_irt import IRT

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TEMPORARY LOAD FROM LOCAL
    response_matrix = torch.load("data/correctness_matrix.pt").to(device)
    response_time_matrix = torch.load("data/time_matrix.pt").to(device)
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts
    
    ##### TEMPORARY FIX -- REMOVE LATER #####
    accept_idxs = []
    for idx, row in enumerate(response_time_matrix):
        if row.unique().shape[0] > 1:
            accept_idxs.append(idx)
            
    response_matrix = response_matrix[accept_idxs]
    response_time_matrix = response_time_matrix[accept_idxs]
    response_time_matrix[response_time_matrix==2314] = -1
    #########################################
    
    irt_model = IRT(D=1, PL=1, low_rank_constraint="distinctGP", device=device)
    
    irt_model.fit(
        method="ess",
        max_epoch=10000,
        response_matrix=response_matrix,
        response_time_matrix=response_time_matrix,
        embedding=None,
        model_features=None
    )
    
    print("Saving model...")
    torch.save(irt_model.ability, "data/ability.pt")
    torch.save(irt_model.difficulty, "data/difficulty.pt")
