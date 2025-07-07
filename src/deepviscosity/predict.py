import os
import click
import numpy as np
import pandas as pd
import random
import joblib
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam # Use tensorflow.keras.optimizers.Adam
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Suppress TensorFlow warnings and disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Ensure consistent results for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# Disable TensorFlow 2.x behavior if using older Keras/TF 1.x models
# This might be necessary if the models were trained with TensorFlow 1.x or older Keras versions
# If you encounter issues, you might need to adjust this based on your TensorFlow/Keras environment.
try:
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.enable_eager_execution()
except AttributeError:
    print("TensorFlow v2 compatibility functions not available, assuming TensorFlow 2.x behavior.")
    print("If models fail to load, consider checking your TensorFlow/Keras versions.")


# Function to preprocess sequences (source: https://github.com/Lailabcode/DeepSCM/blob/main/deepscm-master/seq_preprocessing.py)
def seq_preprocessing(infile_H_path, infile_L_path, outfile_path):
    """
    Preprocesses aligned heavy and light chain sequences into a combined format.

    Args:
        infile_H_path (str): Path to the aligned heavy chain CSV file.
        infile_L_path (str): Path to the aligned light chain CSV file.
        outfile_path (str): Path for the output combined sequence text file.
    """
    infile_H = pd.read_csv(infile_H_path)
    infile_L = pd.read_csv(infile_L_path)

    # Define inclusion lists for heavy and light chain positions
    # These lists define the order and specific positions to extract
    H_inclusion_list = ['1','2','3','4','5','6','7','8','9','10',
                        '11','12','13','14','15','16','17','18','19','20',
                        '21','22','23','24','25','26','27','28','29','30',
                        '31','32','33','34','35','36','37','38','39','40',
                        '41','42','43','44','45','46','47','48','49','50',
                        '51','52','53','54','55','56','57','58','59','60',
                        '61','62','63','64','65','66','67','68','69','70',
                        '71','72','73','74','75','76','77','78','79','80',
                        '81','82','83','84','85','86','87','88','89','90',
                        '91','92','93','94','95','96','97','98','99','100',
                        '101','102','103','104','105','106','107','108','109','110',
                        '111','111A','111B','111C','111D','111E','111F','111G','111H',
                        '112I','112H','112G','112F','112E','112D','112C','112B','112A','112',
                        '113','114','115','116','117','118','119','120',
                        '121','122','123','124','125','126','127','128']

    L_inclusion_list = ['1','2','3','4','5','6','7','8','9','10',
                        '11','12','13','14','15','16','17','18','19','20',
                        '21','22','23','24','25','26','27','28','29','30',
                        '31','32','33','34','35','36','37','38','39','40',
                        '41','42','43','44','45','46','47','48','49','50',
                        '51','52','53','54','55','56','57','58','59','60',
                        '61','62','63','64','65','66','67','68','69','70',
                        '71','72','73','74','75','76','77','78','79','80',
                        '81','82','83','84','85','86','87','88','89','90',
                        '91','92','93','94','95','96','97','98','99','100',
                        '101','102','103','104','105','106','107','108','109','110',
                        '111','112','113','114','115','116','117','118','119','120',
                        '121','122','123','124','125','126','127']

    # Programmatically generate dictionaries for mapping positions to array indices
    # This ensures consistency between the inclusion lists and the dictionaries
    H_dict = {pos: idx for idx, pos in enumerate(H_inclusion_list)}
    L_dict = {pos: idx for idx, pos in enumerate(L_inclusion_list)}

    N_mAbs = len(infile_H["Id"])

    with open(outfile_path, "w") as outfile:
        for i in range(N_mAbs):
            # Initialize temporary sequence arrays with gaps based on expected lengths
            H_tmp = ['-'] * len(H_inclusion_list)
            L_tmp = ['-'] * len(L_inclusion_list)

            # Populate heavy chain sequence based on inclusion list and available columns
            for col in infile_H.columns:
                if col in H_dict: # Check if column is in our mapping dictionary
                    H_tmp[H_dict[col]] = infile_H.iloc[i][col]
            # Populate light chain sequence based on inclusion list and available columns
            for col in infile_L.columns:
                if col in L_dict: # Check if column is in our mapping dictionary
                    L_tmp[L_dict[col]] = infile_L.iloc[i][col]

            aa_string = ''
            for aa in H_tmp + L_tmp:
                aa_string += aa
            outfile.write(infile_H.iloc[i, 0] + " " + aa_string)
            outfile.write("\n")

# Function to load input data from the preprocessed sequence file
def load_input_data(filename):
    """
    Loads antibody names and sequences from a preprocessed text file.

    Args:
        filename (str): Path to the preprocessed sequence text file.

    Returns:
        tuple: A tuple containing two lists: names and sequences.
    """
    name_list = []
    seq_list = []
    with open(filename) as datafile:
        for line in datafile:
            line = line.strip().split()
            name_list.append(line[0])
            seq_list.append(line[1])
    return name_list, seq_list

# Function for one-hot encoding of amino acid sequences
def one_hot_encoder(s):
    """
    Performs one-hot encoding for an amino acid sequence.

    Args:
        s (str): The amino acid sequence.

    Returns:
        numpy.ndarray: The one-hot encoded representation of the sequence.
    """
    # Mapping of amino acids and gap to integer indices
    d = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
         'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20}

    x = np.zeros((len(d), len(s)))
    # Set 1 at the corresponding amino acid index for each position
    x[[d[c] for c in s], range(len(s))] = 1
    return x

# Main command-line interface function using click
@click.command()
@click.option('--input_csv', required=True, help='Path to the input CSV file (e.g., DeepViscosity_input.csv).')
@click.option('--output_csv', required=True, help='Path to the output CSV file or directory where predictions will be saved.')
def main(input_csv, output_csv):
    """
    Predicts DeepViscosity classes for antibody sequences from an input CSV file.
    Intermediate files are stored in the same directory as the output CSV.
    """
    # Determine the actual output file path
    if os.path.isdir(output_csv):
        # If output_csv is a directory, append a default filename
        output_filename = "DeepViscosity_predictions.csv"
        output_csv_filepath = os.path.join(output_csv, output_filename)
        output_dir = output_csv # The user provided a directory, so use it as the output_dir
        print(f"Output path provided is a directory. Saving results to: {output_csv_filepath}")
    else:
        # If output_csv is a file path, extract the directory
        output_csv_filepath = output_csv
        output_dir = os.path.dirname(output_csv_filepath)
        if not output_dir: # Handle cases where only a filename is given (e.g., "output.csv")
            output_dir = "." # Use current directory if no directory is specified

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 현재 스크립트의 절대 경로를 얻고, 프로젝트의 루트 디렉토리를 계산합니다.
    # predict.py는 src/deepviscosity/ 안에 있으므로, 두 레벨 위가 루트입니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    print(f"Loading dataset from: {input_csv}")
    try:
        dataset = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error loading input CSV: {e}")
        return

    name = dataset['Name'].to_list()
    Heavy_seq = dataset['Heavy_Chain'].to_list()
    Light_seq = dataset['Light_Chain'].to_list()

    # Convert to fasta files for ANARCI
    print("Converting sequences to FASTA format...")
    fasta_H_path = os.path.join(output_dir, 'seq_H.fasta')
    fasta_L_path = os.path.join(output_dir, 'seq_L.fasta')

    with open(fasta_H_path, "w") as output_handle:
        for i in range(len(name)):
            record = SeqRecord(
                Seq(Heavy_seq[i]),
                id=name[i],
                name="",
                description="",
            )
            SeqIO.write(record, output_handle, "fasta")

    with open(fasta_L_path, "w") as output_handle:
        for i in range(len(name)):
            record = SeqRecord(
                Seq(Light_seq[i]),
                id=name[i],
                name="",
                description="",
            )
            SeqIO.write(record, output_handle, "fasta")

    # Sequence alignment with ANARCI
    print("Performing sequence alignment with ANARCI...")
    # ANARCI automatically appends _H.csv and _KL.csv to the output base name
    anarci_output_base = os.path.join(output_dir, 'seq_aligned')
    aligned_csv_H_path = anarci_output_base + '_H.csv'
    aligned_csv_KL_path = anarci_output_base + '_KL.csv' # ANARCI outputs KL for light chain

    # Construct ANARCI commands with full paths
    anarci_cmd_H = f'ANARCI -i {fasta_H_path} -o {anarci_output_base} -s imgt -r heavy --csv'
    anarci_cmd_L = f'ANARCI -i {fasta_L_path} -o {anarci_output_base} -s imgt -r light --csv'

    print(f"Executing: {anarci_cmd_H}")
    os.system(anarci_cmd_H)
    print(f"Executing: {anarci_cmd_L}")
    os.system(anarci_cmd_L)

    # Preprocess aligned sequences
    print("Preprocessing aligned sequences...")
    combined_seq_txt_path = os.path.join(output_dir, 'seq_aligned_HL.txt')
    seq_preprocessing(aligned_csv_H_path, aligned_csv_KL_path, combined_seq_txt_path)

    # Load preprocessed sequences
    print("Loading preprocessed sequences for one-hot encoding...")
    name_list, seq_list = load_input_data(combined_seq_txt_path)
    X = seq_list

    # One hot encoding of aligned sequences
    print("Performing one-hot encoding...")
    X = [one_hot_encoder(s=x) for x in X]
    X = np.transpose(np.asarray(X), (0, 2, 1))
    X = np.asarray(X)

    # DeepSP Predictions (models assumed to be in fixed relative paths)
    print("Making DeepSP predictions...")
    # DeepSP 모델의 절대 경로 설정
    deepsp_model_dir = os.path.join(project_root, 'data', 'DeepSP_CNN_model')
    deepsp_descriptors_path = os.path.join(output_dir, 'DeepSP_descriptors.csv') # Path for DeepSP output

    # SAPpos prediction
    try:
        # 파일명에 '_regression_'이 포함된 버전을 사용하도록 수정
        with open(os.path.join(deepsp_model_dir, 'Conv1D_regression_SAPpos.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(deepsp_model_dir, 'Conv1D_regression_SAPpos.h5'))
        loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        sap_pos = loaded_model.predict(X)
    except Exception as e:
        print(f"Error loading or predicting with SAPpos model: {e}")
        return

    # SCMpos prediction
    try:
        # 파일명에 '_regression_'이 포함된 버전을 사용하도록 수정
        with open(os.path.join(deepsp_model_dir, 'Conv1D_regression_SCMpos.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(deepsp_model_dir, 'Conv1D_regression_SCMpos.h5'))
        loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        scm_pos = loaded_model.predict(X)
    except Exception as e:
        print(f"Error loading or predicting with SCMpos model: {e}")
        return

    # SCMneg prediction
    try:
        # 파일명에 '_regression_'이 포함된 버전을 사용하도록 수정
        with open(os.path.join(deepsp_model_dir, 'Conv1D_regression_SCMneg.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(deepsp_model_dir, 'Conv1D_regression_SCMneg.h5'))
        loaded_model.compile(optimizer='adam', loss='mae', metrics=['mae'])
        scm_neg = loaded_model.predict(X)
    except Exception as e:
        print(f"Error loading or predicting with SCMneg model: {e}")
        return


    # Combine DeepSP features and save to CSV
    features = ['Name', 'SAP_pos_CDRH1', 'SAP_pos_CDRH2', 'SAP_pos_CDRH3', 'SAP_pos_CDRL1', 'SAP_pos_CDRL2', 'SAP_pos_CDRL3', 'SAP_pos_CDR', 'SAP_pos_Hv', 'SAP_pos_Lv', 'SAP_pos_Fv',
                'SCM_neg_CDRH1', 'SCM_neg_CDRH2', 'SCM_neg_CDRH3', 'SCM_neg_CDRL1', 'SCM_neg_CDRL2', 'SCM_neg_CDRL3', 'SCM_neg_CDR', 'SCM_neg_Hv', 'SCM_neg_Lv', 'SCM_neg_Fv',
                'SCM_pos_CDRH1', 'SCM_pos_CDRH2', 'SCM_pos_CDRH3', 'SCM_pos_CDRL1', 'SCM_pos_CDRL2', 'SCM_pos_CDRL3', 'SCM_pos_CDR', 'SCM_pos_Hv', 'SCM_pos_Lv', 'SCM_pos_Fv']
    df_deepsp = pd.concat([pd.DataFrame(name_list), pd.DataFrame(sap_pos), pd.DataFrame(scm_neg), pd.DataFrame(scm_pos)], ignore_index=True, axis=1)
    df_deepsp.columns = features
    df_deepsp.to_csv(deepsp_descriptors_path, index=False)
    print(f"DeepSP descriptors saved to: {deepsp_descriptors_path}")

    # DeepViscosity Predictions [ Low viscosity(<=20cps) : 0, High viscosity(>20cps) : 1 ]
    print("Making DeepViscosity predictions...")
    X_deepvis = df_deepsp.iloc[:, 1:]

    # DeepViscosity 스케일러의 절대 경로 설정
    deepviscosity_scaler_path = os.path.join(project_root, 'data', 'DeepViscosity_scaler', 'DeepViscosity_scaler.save')
    try:
        scaler = joblib.load(deepviscosity_scaler_path)
        X_scaled = scaler.transform(X_deepvis.values)
    except FileNotFoundError:
        print(f"Error: DeepViscosity scaler file not found at {deepviscosity_scaler_path}")
        return
    except Exception as e:
        print(f"Error loading or applying DeepViscosity scaler: {e}")
        return

    model_preds = []
    # DeepViscosity ANN 앙상블 모델의 절대 경로 설정
    deepviscosity_ann_models_dir = os.path.join(project_root, 'data', 'DeepViscosity_ANN_ensemble_models')

    for i in range(102):
        file_name = 'ANN_logo_' + str(i)
        json_model_path = os.path.join(deepviscosity_ann_models_dir, file_name + '.json')
        h5_weights_path = os.path.join(deepviscosity_ann_models_dir, file_name + '.h5')

        try:
            with open(json_model_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(h5_weights_path)
            model.compile(optimizer=Adam(learning_rate=0.0001), metrics=['accuracy']) # Use learning_rate instead of lr for newer Adam
            pred = model.predict(X_scaled, verbose=0)
            model_preds.append(pred)
        except FileNotFoundError:
            print(f"Warning: DeepViscosity ANN model file not found for {file_name}. Skipping this model.")
            continue
        except Exception as e:
            print(f"Error loading or predicting with DeepViscosity ANN model {file_name}: {e}. Skipping this model.")
            continue

    if not model_preds:
        print("No DeepViscosity models were successfully loaded or predicted. Cannot make final prediction.")
        return

    # Combine the predictions using majority voting
    final_pred = np.where(np.array(model_preds).mean(axis=0) >= 0.5, 1, 0)

    # Save final DeepViscosity predictions
    df_deepvis = pd.concat([pd.DataFrame(name_list), pd.DataFrame(final_pred)], ignore_index=True, axis=1)
    df_deepvis.columns = ['Name', 'DeepViscosity_classes']
    # Use the determined output_csv_filepath for saving the final result
    df_deepvis.to_csv(output_csv_filepath, index=False)
    print(f"DeepViscosity predictions saved to: {output_csv_filepath}")
    print("Processing complete!")

if __name__ == '__main__':
    main()