# Import libraries
import os
import subprocess
import random
import logging # Added for logging

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam # Corrected import
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# --- Global Constants and Configuration ---
# Suppress TensorFlow/oneDNN informational messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Script directory and base data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
INPUT_DIR = os.path.join(BASE_DATA_DIR, 'input')
DEEPSP_MODEL_DIR = os.path.join(BASE_DATA_DIR, 'DeepSP_CNN_model')
DEEPVISCOSITY_SCALER_DIR = os.path.join(BASE_DATA_DIR, 'DeepViscosity_scaler')
DEEPVISCOSITY_MODEL_DIR = os.path.join(BASE_DATA_DIR, 'DeepViscosity_ANN_ensemble_models')

# Temporary file paths (created within the script's directory)
SEQ_H_FASTA_PATH = os.path.join(SCRIPT_DIR, 'seq_H.fasta')
SEQ_L_FASTA_PATH = os.path.join(SCRIPT_DIR, 'seq_L.fasta')
ANARCI_H_OUT_PATH = os.path.join(SCRIPT_DIR, 'seq_aligned_H.csv')
ANARCI_L_OUT_PATH = os.path.join(SCRIPT_DIR, 'seq_aligned_KL.csv')
ALIGNED_HL_PATH = os.path.join(INPUT_DIR, 'seq_aligned_HL.txt') # Output to data/input

# Output file paths
DEEPSP_DESCRIPTORS_PATH = os.path.join(BASE_DATA_DIR, 'DeepSP_descriptors.csv')
DEEPVISCOSITY_CLASSES_PATH = os.path.join(BASE_DATA_DIR, 'DeepViscosity_classes.csv')

# ANARCI base output name (without extension)
ANARCI_BASE_OUT_NAME = os.path.join(SCRIPT_DIR, 'seq_aligned')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def create_fasta_file(sequences: list[str], names: list[str], file_path: str):
    """Creates a FASTA file from sequences and names."""
    logging.info(f"Creating FASTA file: {file_path}")
    with open(file_path, "w") as output_handle:
        for i, seq_str in enumerate(sequences):
            record = SeqRecord(Seq(seq_str), id=names[i], name="", description="")
            SeqIO.write(record, output_handle, "fasta")
    logging.info(f"FASTA file created successfully: {file_path}")

def run_anarci(input_fasta_path: str, output_base_name: str, chain_type: str):
    """Runs ANARCI alignment."""
    logging.info(f"Running ANARCI for {chain_type} chain: {input_fasta_path}")
    try:
        subprocess.run(
            ['ANARCI', '-i', input_fasta_path, '-o', output_base_name, '-s', 'imgt', '-r', chain_type, '--csv'],
            check=True, capture_output=True, text=True
        )
        logging.info(f"ANARCI alignment successful for {chain_type} chain.")
    except subprocess.CalledProcessError as e:
        logging.error(f"ANARCI failed for {chain_type} chain. Error: {e.stderr}")
        raise  # Re-raise the exception to stop the script

def preprocess_aligned_sequences(h_aligned_path: str, l_aligned_path: str, outfile_path: str):
    """
    Performs sequence alignment preprocessing based on DeepSCM's method.
    Source: https://github.com/Lailabcode/DeepSCM/blob/main/deepscm-master/seq_preprocessing.py
    """
    logging.info("Starting sequence preprocessing...")
    try:
        infile_H = pd.read_csv(h_aligned_path)
        infile_L = pd.read_csv(l_aligned_path)
    except FileNotFoundError as e:
        logging.error(f"Error reading ANARCI output files: {e}")
        raise

    with open(outfile_path, "w") as outfile:
        H_inclusion_list = [str(i) for i in range(1, 129)] + \
                           ['111A','111B','111C','111D','111E','111F','111G','111H',
                            '112I','112H','112G','112F','112E','112D','112C','112B','112A']
        L_inclusion_list = [str(i) for i in range(1, 128)]

        # Simplified H_dict and L_dict creation
        h_pos_map = {
            **{str(i): i-1 for i in range(1, 112)},
            '111A':111,'111B':112,'111C':113,'111D':114,'111E':115,'111F':116,'111G':117,'111H':118,
            '112I':119,'112H':120,'112G':121,'112F':122,'112E':123,'112D':124,'112C':125,'112B':126,'112A':127,
            **{str(i): i+16 for i in range(112, 129)} # Adjusted mapping for 112 onwards
        }
        # Correcting H_dict for positions like '112', '113' etc.
        h_idx = 110 # last index from 1-111 range
        for i in range(1, 10): # 111A-H and 112A-I
            for letter_code in ['', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                if f"111{letter_code}" in H_inclusion_list and f"111{letter_code}" not in h_pos_map :
                    h_pos_map[f"111{letter_code}"] = h_idx
                    h_idx+=1
                if f"112{letter_code}" in H_inclusion_list and f"112{letter_code}" not in h_pos_map :
                     h_pos_map[f"112{letter_code}"] = h_idx
                     h_idx+=1
        for i in range(113,129): #113-128
            if str(i) in H_inclusion_list and str(i) not in h_pos_map:
                h_pos_map[str(i)] = h_idx
                h_idx+=1


        l_pos_map = {str(i): i-1 for i in range(1, 128)}

        N_mAbs = len(infile_H["Id"])
        for i in range(N_mAbs):
            H_tmp = 145*['-'] # Max length for H chain IMGT numbering can be around 140-150
            L_tmp = 127*['-'] # Max length for L chain IMGT numbering
            for col in infile_H.columns:
                if col in H_inclusion_list and col in h_pos_map: # Check if col is in h_pos_map
                    pos_idx = h_pos_map[col]
                    if 0 <= pos_idx < len(H_tmp):
                         H_tmp[pos_idx]=infile_H.iloc[i][col]
                    else:
                        logging.warning(f"Index {pos_idx} for H-chain column {col} is out of bounds for H_tmp (len {len(H_tmp)}). Skipping.")

            for col in infile_L.columns:
                if col in L_inclusion_list and col in l_pos_map: # Check if col is in l_pos_map
                    pos_idx = l_pos_map[col]
                    if 0 <= pos_idx < len(L_tmp):
                        L_tmp[pos_idx]=infile_L.iloc[i][col]
                    else:
                        logging.warning(f"Index {pos_idx} for L-chain column {col} is out of bounds for L_tmp (len {len(L_tmp)}). Skipping.")


            aa_string = "".join(H_tmp + L_tmp)
            outfile.write(f"{infile_H.iloc[i,0]} {aa_string}\n")
    logging.info("Sequence preprocessing finished.")


def load_aligned_data(filename: str) -> tuple[list[str], list[str]]:
    """Loads aligned sequence data from a file."""
    logging.info(f"Loading aligned data from: {filename}")
    name_list, seq_list = [], []
    try:
        with open(filename) as datafile:
            for line in datafile:
                parts = line.strip().split()
                if len(parts) == 2: # Ensure correct format
                    name_list.append(parts[0])
                    seq_list.append(parts[1])
                else:
                    logging.warning(f"Skipping malformed line in {filename}: {line.strip()}")
    except FileNotFoundError:
        logging.error(f"Aligned data file not found: {filename}")
        raise
    return name_list, seq_list

def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """One-hot encodes a single amino acid sequence."""
    aa_dict = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,
               'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,
               'W':18,'Y':19,'-':20, 'X':20} # Added 'X' to handle unknown/other AAs like '-'

    # Replace any non-standard amino acids not in dict with 'X' (which maps to '-')
    processed_sequence = "".join([s if s in aa_dict else 'X' for s in sequence])

    encoded_seq = np.zeros((len(aa_dict)-1, len(processed_sequence))) # Exclude 'X' itself from dimension if it's just a placeholder

    for i, char_s in enumerate(processed_sequence):
        if char_s in aa_dict and aa_dict[char_s] < (len(aa_dict)-1): # Ensure index is within bounds
             encoded_seq[aa_dict[char_s], i] = 1
    return encoded_seq


def predict_deepsp_features(X_encoded: np.ndarray, model_type: str) -> np.ndarray:
    """Loads a DeepSP model and predicts features."""
    logging.info(f"Predicting DeepSP features for model type: {model_type}")
    json_path = os.path.join(DEEPSP_MODEL_DIR, f'Conv1D_regression{model_type.upper()}.json')
    weights_path = os.path.join(DEEPSP_MODEL_DIR, f'Conv1D_regression_{model_type.lower()}.h5')

    try:
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        predictions = model.predict(X_encoded)
        logging.info(f"DeepSP {model_type} prediction successful.")
        return predictions
    except Exception as e:
        logging.error(f"Error during DeepSP {model_type} prediction: {e}")
        raise

def predict_deepviscosity(df_deepsp_features: pd.DataFrame, scaler_path: str, model_dir: str) -> pd.DataFrame:
    """Predicts DeepViscosity classes using an ensemble of ANN models."""
    logging.info("Starting DeepViscosity prediction...")
    X_features = df_deepsp_features.iloc[:, 1:].values # Use .values to get numpy array

    try:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_features)
    except FileNotFoundError:
        logging.error(f"Scaler file not found: {scaler_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading or applying scaler: {e}")
        raise

    model_preds = []
    num_models = 102 # Assuming 102 models as in the original script
    logging.info(f"Loading {num_models} ANN ensemble models for DeepViscosity...")
    for i in range(num_models):
        file_name_prefix = f'ANN_logo_{i}'
        model_json_path = os.path.join(model_dir, f'{file_name_prefix}.json')
        model_h5_path = os.path.join(model_dir, f'{file_name_prefix}.h5')

        try:
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_h5_path)
            model.compile(optimizer=Adam(learning_rate=0.0001), metrics=['accuracy']) # Use learning_rate
            pred = model.predict(X_scaled, verbose=0)
            model_preds.append(pred)
        except Exception as e:
            logging.error(f"Error loading or predicting with ANN model {file_name_prefix}: {e}")
            # Optionally, decide whether to skip this model or raise the error
            # For now, let's re-raise to be safe
            raise

    logging.info("Combining predictions using majority voting.")
    final_pred = np.where(np.array(model_preds).mean(axis=0) >= 0.5, 1, 0)
    logging.info("DeepViscosity prediction finished.")
    return final_pred

def cleanup_temp_files(files_to_remove: list[str]):
    """Removes specified temporary files."""
    logging.info("Cleaning up temporary files...")
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Removed temporary file: {file_path}")
        except OSError as e:
            logging.warning(f"Error removing temporary file {file_path}: {e}")
    # Clean up other ANARCI outputs (more robustly)
    for item in os.listdir(SCRIPT_DIR):
        if item.startswith("seq_aligned") and (item.endswith(".al") or item.endswith(".ali") or item.endswith(".pdb")):
            try:
                os.remove(os.path.join(SCRIPT_DIR, item))
                logging.info(f"Removed ANARCI output: {item}")
            except OSError as e:
                 logging.warning(f"Error removing ANARCI output {item}: {e}")


# --- Main Execution ---
def main():
    logging.info("Starting DeepViscosity predictor script.")

    # Ensure input directory exists
    os.makedirs(INPUT_DIR, exist_ok=True)

    # 1. Import dataset
    logging.info(f"Importing dataset from: {dataset_path}")
    try:
        dataset = pd.read_csv(dataset_path)
    except FileNotFoundError:
        logging.error(f"Input dataset not found: {dataset_path}")
        return # Exit if input file is missing

    names = dataset['Name'].to_list()
    heavy_seqs = dataset['Heavy_Chain'].to_list()
    light_seqs = dataset['Light_Chain'].to_list()

    # 2. Create FASTA files
    create_fasta_file(heavy_seqs, names, SEQ_H_FASTA_PATH)
    create_fasta_file(light_seqs, names, SEQ_L_FASTA_PATH)

    # 3. Run ANARCI
    run_anarci(SEQ_H_FASTA_PATH, ANARCI_BASE_OUT_NAME, 'heavy')
    run_anarci(SEQ_L_FASTA_PATH, ANARCI_BASE_OUT_NAME, 'light')

    # 4. Preprocess aligned sequences
    preprocess_aligned_sequences(ANARCI_H_OUT_PATH, ANARCI_L_OUT_PATH, ALIGNED_HL_PATH)

    # 5. Load aligned data and one-hot encode
    loaded_names, loaded_seqs = load_aligned_data(ALIGNED_HL_PATH)
    logging.info("One-hot encoding sequences...")
    X_encoded_list = [one_hot_encode_sequence(s) for s in loaded_seqs]

    # Ensure all encoded sequences have the same length for stacking
    # This might involve padding or truncating, depending on the model's requirements.
    # For now, assuming models handle variable length or sequences are already standardized.
    # If padding is needed, find max_len and pad.
    # max_len = max(x.shape[1] for x in X_encoded_list)
    # X_padded = [np.pad(x, ((0,0), (0, max_len - x.shape[1])), 'constant') for x in X_encoded_list]
    # X_one_hot = np.array(X_padded)
    # The original script used np.transpose(np.asarray(X), (0, 2, 1))
    # which implies sequences are of same length after one-hot encoding and then transposed.
    # Let's replicate that structure, assuming one_hot_encoder produces (num_features, seq_len)
    # and we need (batch_size, seq_len, num_features)

    # Transpose each sequence from (num_features, seq_len) to (seq_len, num_features)
    X_transposed_list = [x.T for x in X_encoded_list]

    # Pad sequences to the same length if necessary (common for CNNs)
    max_seq_len = 0
    if X_transposed_list:
        max_seq_len = max(x.shape[0] for x in X_transposed_list)

    X_padded_list = []
    num_features = 0
    if X_transposed_list:
        num_features = X_transposed_list[0].shape[1] # Should be consistent (20 or 21)

    for x_t in X_transposed_list:
        padding_length = max_seq_len - x_t.shape[0]
        if padding_length > 0:
            # Pad with zeros (or a specific padding vector if required by model)
            padding_array = np.zeros((padding_length, num_features))
            x_padded = np.vstack((x_t, padding_array))
        else:
            x_padded = x_t
        X_padded_list.append(x_padded)

    if not X_padded_list:
        logging.error("No sequences to process after one-hot encoding and padding.")
        return

    X_one_hot = np.asarray(X_padded_list)


    # 6. DeepSP Predictions
    sap_pos = predict_deepsp_features(X_one_hot, 'SAPpos')
    scm_pos = predict_deepsp_features(X_one_hot, 'SCMpos')
    scm_neg = predict_deepsp_features(X_one_hot, 'SCMneg')

    # Create DeepSP features DataFrame
    deepsp_features_df = pd.concat([
        pd.DataFrame(loaded_names, columns=['Name']),
        pd.DataFrame(sap_pos), pd.DataFrame(scm_neg), pd.DataFrame(scm_pos)
    ], axis=1)

    # Define column names for the DeepSP features DataFrame
    # (Ensure this matches the number of columns from the concatenated predictions)
    num_sap_features = sap_pos.shape[1] if sap_pos.ndim > 1 else 1
    num_scm_neg_features = scm_neg.shape[1] if scm_neg.ndim > 1 else 1
    num_scm_pos_features = scm_pos.shape[1] if scm_pos.ndim > 1 else 1

    deepsp_cols = ['Name'] + \
                  [f'SAP_pos_{i+1}' for i in range(num_sap_features)] + \
                  [f'SCM_neg_{i+1}' for i in range(num_scm_neg_features)] + \
                  [f'SCM_pos_{i+1}' for i in range(num_scm_pos_features)]

    if len(deepsp_cols) == len(deepsp_features_df.columns):
        deepsp_features_df.columns = deepsp_cols
    else:
        logging.warning("Mismatch in DeepSP feature columns count. Using default numbered columns.")
        # Fallback to default numbered columns if there's a mismatch
        # This part might need adjustment based on the actual output shape of predict_deepsp_features

    deepsp_features_df.to_csv(DEEPSP_DESCRIPTORS_PATH, index=False)
    logging.info(f"DeepSP descriptors saved to: {DEEPSP_DESCRIPTORS_PATH}")

    # 7. DeepViscosity Predictions
    final_predictions = predict_deepviscosity(deepsp_features_df, DEEPVISCOSITY_SCALER_DIR + "/DeepViscosity_scaler.save", DEEPVISCOSITY_MODEL_DIR)

    df_deepvis = pd.DataFrame({'Name': loaded_names, 'DeepViscosity_classes': final_predictions.flatten()})
    df_deepvis.to_csv(DEEPVISCOSITY_CLASSES_PATH, index=False)
    logging.info(f"DeepViscosity classes saved to: {DEEPVISCOSITY_CLASSES_PATH}")

    # 8. Cleanup temporary files
    files_to_clean = [SEQ_H_FASTA_PATH, SEQ_L_FASTA_PATH, ANARCI_H_OUT_PATH, ANARCI_L_OUT_PATH]
    # ALIGNED_HL_PATH is in INPUT_DIR, decide if it's temporary or intermediate output.
    # If it's purely temporary for this script run, add it to files_to_clean.
    # For now, assuming it might be a useful intermediate, so not deleting.
    cleanup_temp_files(files_to_clean)

    logging.info("DeepViscosity predictor script finished successfully.")

if __name__ == '__main__':
    main()
# --- Old Code (commented out for reference, to be removed after refactoring) ---
# # Import dataset
# dataset = pd.read_csv('../data/input/DeepViscosity_input.csv') # replace with your csv file, see format in DeepViscosity_input.csv file
# name = dataset['Name'].to_list()
# Heavy_seq = dataset['Heavy_Chain'].to_list()
# Light_seq = dataset['Light_Chain'].to_list()

# # convert to fasta file
# file_out='seq_H.fasta'
# with open(file_out, "w") as output_handle:
#   for i in range(len(name)):
#     seq_name = name[i]
#     seq = Heavy_seq[i]
#     record = SeqRecord(
#     Seq(seq),
#     id=seq_name,
#     name="",
#     description="",
#     )
#     SeqIO.write(record, output_handle, "fasta")

# file_out='seq_L.fasta'
# with open(file_out, "w") as output_handle:
#   for i in range(len(name)):
#     seq_name = name[i]
#     seq = Light_seq[i]
#     record = SeqRecord(
#     Seq(seq),
#     id=seq_name,
#     name="",
#     description="",
#     )
#     SeqIO.write(record, output_handle, "fasta")

# # sequence alignment with ANARCI
# os.system('ANARCI -i seq_H.fasta -o seq_aligned -s imgt -r heavy --csv')
# os.system('ANARCI -i seq_L.fasta -o seq_aligned -s imgt -r light --csv')

# H_aligned = pd.read_csv('../data/input/seq_aligned_H.csv')
# L_aligned = pd.read_csv('../data/input/seq_aligned_KL.csv')

# # sequence alignment - source: # https://github.com/Lailabcode/DeepSCM/blob/main/deepscm-master/seq_preprocessing.py
# def seq_preprocessing():
#   infile_H = pd.read_csv('../data/input/seq_aligned_H.csv')
#   infile_L = pd.read_csv('../data/input/seq_aligned_KL.csv')
#   outfile = open('../data/input/seq_aligned_HL.txt', "w")

#   H_inclusion_list = ['1','2','3','4','5','6','7','8','9','10', \
# ... (rest of the old code)