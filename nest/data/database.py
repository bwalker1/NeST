
import pandas as pd
import numpy as np


def load_database():
    # set up to run the R code
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        import anndata2ri
        anndata2ri.activate()
        robjects = robjects

        importr("CellChat")
    except ImportError:
        raise ImportError("Ensure rpy2 and CellChat are installed correctly.")

    cellchat_db = robjects.r['CellChatDB.mouse'][0]
    interaction_cellchat = pd.DataFrame(cellchat_db).fillna('').transpose()
    interaction_cellchat.columns = list(cellchat_db.colnames)

    # Restructure column names to desired format
    is_secreted = interaction_cellchat["annotation"] == "Secreted Signaling"
    not_ecm = np.logical_not(interaction_cellchat["annotation"] == "ECM-Receptor")
    interaction_cellchat = interaction_cellchat[["interaction_name", "interaction_name_2", "ligand", "receptor",
                                                 "pathway_name"]].copy()
    interaction_cellchat["secreted"] = is_secreted

    interaction_cellchat = interaction_cellchat[not_ecm].reset_index(drop=True)
    interaction = interaction_cellchat

    return interaction


if __name__=="__main__":
    interaction = load_database()
    print(interaction)