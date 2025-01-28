#%%
from flask import Flask, render_template,request
import pickle
import numpy as np
std_model = pickle.load(open('models/std.sav', 'rb'))
pca_model = pickle.load(open('models/pca.sav', 'rb'))
ann_model = pickle.load(open( 'models/ann.sav', 'rb'))

app = Flask(__name__)
#%% 

def calculate_features_for_sequence(user_sequence):
    """
    Prompts the user to input an amino acid sequence,

    """
    # Get the sequence from the user
    # user_sequence = input("Enter an amino acid sequence (with valid animo acid notations): ").upper()

    # Validate the input (ensure it contains only valid amino acids)
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    if not set(user_sequence).issubset(valid_amino_acids):
        print("Error: Invalid sequence. Please enter a valid amino acid sequence.")
        return
    from rpy2.robjects import r
    from rpy2.robjects.packages import importr
    from rpy2.robjects.vectors import StrVector
    utils = importr('utils')
    try:
        protr = importr('protr')
    except:
        print("Installing protr package...")
        utils.install_packages(StrVector(['protr']))
        protr = importr('protr')
    # Pass the sequence to R and calculate AAC
    r_code = f"""
    library(protr)
    user_seq <- "{user_sequence}"
    aac <- extractAAC(user_seq)
    dc <- extractDC(user_seq)
    tc <- extractTC(user_seq)
    mbc <- extractMoreauBroto(user_seq)
    mc <- extractMoran(user_seq)
    geary <- extractGeary(user_seq)
    c <- extractCTDC(user_seq)
    t <- extractCTDT(user_seq)
    d <- extractCTDD(user_seq)
    conjoint_triad <- extractCTriad(user_seq)
    socp <- extractSOCN(user_seq)
    qso <- extractQSO(user_seq)
    paac <- extractPAAC(user_seq)
    apaac <- extractAPAAC(user_seq)
    blosum <- extractBLOSUM(user_seq, k=5, lag=7)
    descscales <- extractDescScales(user_seq, propmat="AATopo", pc=20, lag=7)
    descscales_aamoe2d <- extractDescScales(user_seq, propmat="AAMOE2D", pc=20, lag=7)
    descscales_aamoe3d <- extractDescScales(user_seq, propmat="AAMOE3D", pc=20, lag=7)
    descscales_aamolprop <- extractDescScales(user_seq, propmat="AAMolProp", pc=20, lag=7)
    scale_based_des <- Reduce(function(x, y) cbind(x, y),
                       list(descscales_aamolprop,
                            descscales_aamoe2d,
                            descscales_aamoe3d,
                            descscales))
    aac_mat <- matrix(aac, nrow = 1)
    colnames(aac_mat) <- names(aac)

    dc_mat <- matrix(dc, nrow = 1)
    colnames(dc_mat) <- names(dc)

    tc_mat <- matrix(tc, nrow = 1)
    colnames(tc_mat) <- names(tc)

    mbc_mat <- matrix(mbc, nrow = 1)
    colnames(mbc_mat) <- names(mbc)

    mc_mat <- matrix(mc, nrow = 1)
    colnames(mc_mat) <- names(mc)

    geary_mat <- matrix(geary, nrow = 1)
    colnames(geary_mat) <- names(geary)

    c_mat <- matrix(c, nrow = 1)
    colnames(c_mat) <- names(c)

    t_mat <- matrix(t, nrow = 1)
    colnames(t_mat) <- names(t)

    d_mat <- matrix(d, nrow = 1)
    colnames(d_mat) <- names(d)

    conjoint_triad_mat <- matrix(conjoint_triad, nrow = 1)
    colnames(conjoint_triad_mat) <- names(conjoint_triad)

    socp_mat <- matrix(socp, nrow = 1)
    colnames(socp_mat) <- names(socp)

    qso_mat <- matrix(qso, nrow = 1)
    colnames(qso_mat) <- names(qso)

    paac_mat <- matrix(paac, nrow = 1)
    colnames(paac_mat) <- names(paac)

    apaac_mat <- matrix(apaac, nrow = 1)
    colnames(apaac_mat) <- names(apaac)

    blosum_mat <- matrix(blosum, nrow = 1)
    colnames(blosum_mat) <- names(blosum)

    descscales_mat <- matrix(descscales, nrow = 1)
    colnames(descscales_mat) <- names(descscales)

    descscales_aamoe2d_mat <- matrix(descscales_aamoe2d, nrow = 1)
    colnames(descscales_aamoe2d_mat) <- names(descscales_aamoe2d)

    descscales_aamoe3d_mat <- matrix(descscales_aamoe3d, nrow = 1)
    colnames(descscales_aamoe3d_mat) <- names(descscales_aamoe3d)

    descales_aamolprop_mat <- matrix(descscales_aamolprop, nrow = 1)
    colnames(descales_aamolprop_mat) <- names(descscales_aamolprop)

    all_features <- Reduce(function(i, j) cbind(i, j),
                       list(aac_mat, dc_mat, tc_mat, mbc_mat, mc_mat, geary_mat,
                            c_mat, t_mat, d_mat, conjoint_triad_mat,
                            socp_mat, qso_mat, paac_mat, apaac_mat, blosum_mat,
                            descscales_mat, descscales_aamoe2d_mat, descscales_aamoe3d_mat,
                            descales_aamolprop_mat))
    """
    r(r_code)
    
    all_features_result = list(r("all_features"))
    return all_features_result
#%%
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        
        # Get the selected dropdown value

        # Get the multi-line input data from the form
        data = request.form.get("user_input")
        
 
        # Create a DataFrame from the list of rows
        # try:
        # data = str(data)

        features = calculate_features_for_sequence(data) 
        x =  np.array(features)
        x = x.reshape(1,-1)
        dt_std = std_model.transform(x)
        dt_pca = pca_model.transform(dt_std)
        out_put = ann_model.predict(dt_pca)
        if out_put == 1:
            final_out ="Sequence is disease resistant"
        elif out_put == 0:
            final_out ="Sequence is not disease resistant"
        # except:
            # return render_template("index.html",err=True)
            

        # Return the DataFrame as HTML
        return render_template("index.html",seq_info=str(data),
                               n_var=final_out)
    else:
        return render_template("index.html", df=None)
if __name__=="__main__":
    app.run(debug=True)