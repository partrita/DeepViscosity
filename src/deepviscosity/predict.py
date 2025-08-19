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
def seq_preprocessing(infile_H_path, infile_L_path, outfile_path, chunk_size=50000):
    """
    Preprocesses aligned heavy and light chain sequences into a combined format.
    Optimized version using chunked processing for large files.
    """
    # Define inclusion lists (keep existing lists)
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
    H_dict = {pos: idx for idx, pos in enumerate(H_inclusion_list)}
    L_dict = {pos: idx for idx, pos in enumerate(L_inclusion_list)}
    
    # Get total number of rows for progress tracking
    print("Counting total sequences...")
    with open(infile_H_path, 'r') as f:
        total_rows = sum(1 for line in f) - 1  # Subtract header
    
    print(f"Processing {total_rows} sequences in chunks of {chunk_size}...")
    
    # Process files in chunks to reduce memory usage
    processed_rows = 0
    
    with open(outfile_path, "w") as outfile:
        # Read heavy chain file in chunks
        H_reader = pd.read_csv(infile_H_path, chunksize=chunk_size)
        L_reader = pd.read_csv(infile_L_path, chunksize=chunk_size)
        
        for chunk_idx, (H_chunk, L_chunk) in enumerate(zip(H_reader, L_reader)):
            print(f"Processing chunk {chunk_idx + 1} ({processed_rows + 1}-{processed_rows + len(H_chunk)} sequences)")
            
            # Get relevant columns only to speed up processing
            H_relevant_cols = [col for col in H_chunk.columns if col in H_dict or col == 'Id']
            L_relevant_cols = [col for col in L_chunk.columns if col in L_dict or col == 'Id']
            
            H_chunk_filtered = H_chunk[H_relevant_cols]
            L_chunk_filtered = L_chunk[L_relevant_cols]
            
            # Process each sequence in the chunk
            for i in range(len(H_chunk_filtered)):
                # Initialize temporary sequence arrays with gaps
                H_tmp = ['-'] * len(H_inclusion_list)
                L_tmp = ['-'] * len(L_inclusion_list)

                # Populate heavy chain sequence
                for col in H_chunk_filtered.columns:
                    if col in H_dict:
                        H_tmp[H_dict[col]] = H_chunk_filtered.iloc[i][col]
                
                # Populate light chain sequence
                for col in L_chunk_filtered.columns:
                    if col in L_dict:
                        L_tmp[L_dict[col]] = L_chunk_filtered.iloc[i][col]

                # Combine sequences
                aa_string = ''.join(H_tmp + L_tmp)
                outfile.write(f"{H_chunk_filtered.iloc[i, 0]} {aa_string}\n")
            
            processed_rows += len(H_chunk)
            
            # Progress update
            if chunk_idx % 5 == 0:
                print(f"Progress: {processed_rows}/{total_rows} sequences ({100*processed_rows/total_rows:.1f}%)")

    print("Sequence preprocessing completed.")

# Function to load input data from the preprocessed sequence file
def load_input_data(filename):
    """
    Loads antibody names and sequences from a preprocessed text file.
    Optimized for large files.

    Args:
        filename (str): Path to the preprocessed sequence text file.

    Returns:
        tuple: A tuple containing two lists: names and sequences.
    """
    print("Loading preprocessed sequences...")
    name_list = []
    seq_list = []
    
    # Count total lines for progress tracking
    with open(filename, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Loading {total_lines} sequences...")
    
    with open(filename) as datafile:
        for i, line in enumerate(datafile):
            line = line.strip().split(maxsplit=1)  # Split only on first space
            name_list.append(line[0])
            seq_list.append(line[1])
            
            # Progress update
            if (i + 1) % 100000 == 0:
                print(f"Loaded {i + 1}/{total_lines} sequences ({100*(i+1)/total_lines:.1f}%)")
    
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

def batch_one_hot_encode(seq_list, batch_size=10000):
    """
    Performs one-hot encoding for multiple sequences in batches.
    
    Args:
        seq_list (list): List of amino acid sequences.
        batch_size (int): Number of sequences to process at once.
    
    Returns:
        numpy.ndarray: The one-hot encoded representation of all sequences.
    """
    print(f"Performing one-hot encoding for {len(seq_list)} sequences in batches of {batch_size}...")
    
    n_sequences = len(seq_list)
    n_batches = (n_sequences + batch_size - 1) // batch_size
    
    encoded_batches = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_sequences)
        
        print(f"Encoding batch {batch_idx + 1}/{n_batches} (sequences {start_idx + 1}-{end_idx})")
        
        # Process batch
        batch_sequences = seq_list[start_idx:end_idx]
        batch_encoded = [one_hot_encoder(s) for s in batch_sequences]
        
        # Convert to numpy array and transpose
        batch_array = np.transpose(np.asarray(batch_encoded), (0, 2, 1))
        encoded_batches.append(batch_array)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    # Combine all batches
    print("Combining encoded batches...")
    X = np.vstack(encoded_batches) if len(encoded_batches) > 1 else encoded_batches[0]
    
    return X

# CSV 파일 처리를 위해 chunk 처리 방식 도입
def process_sequences_in_chunks(input_csv, chunk_size=1000):
    """
    Process large CSV files in chunks to reduce memory usage
    """
    chunks = []
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        chunks.append({
            'name': chunk['Name'].tolist(),
            'heavy': chunk['Heavy_Chain'].tolist(),
            'light': chunk['Light_Chain'].tolist()
        })
    return chunks

def process_predictions_in_batches(X, model_dir, batch_size=5000):
    """
    Process predictions in batches to handle large datasets
    """
    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Processing {n_samples} samples in {n_batches} batches of size {batch_size}")
    
    all_predictions = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (samples {start_idx}-{end_idx-1})")
        
        # SAPpos prediction
        try:
            with open(os.path.join(model_dir, 'Conv1D_regression_SAPpos.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(model_dir, 'Conv1D_regression_SAPpos.h5'))
            model.compile(optimizer='adam', loss='mae', metrics=['mae'])
            sap_pos_batch = model.predict(X_batch, verbose=0)
            del model  # Free memory
        except Exception as e:
            print(f"Error with SAPpos model in batch {batch_idx + 1}: {e}")
            return None, None, None
        
        # SCMpos prediction
        try:
            with open(os.path.join(model_dir, 'Conv1D_regression_SCMpos.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(model_dir, 'Conv1D_regression_SCMpos.h5'))
            model.compile(optimizer='adam', loss='mae', metrics=['mae'])
            scm_pos_batch = model.predict(X_batch, verbose=0)
            del model  # Free memory
        except Exception as e:
            print(f"Error with SCMpos model in batch {batch_idx + 1}: {e}")
            return None, None, None
        
        # SCMneg prediction
        try:
            with open(os.path.join(model_dir, 'Conv1D_regression_SCMneg.json'), 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(os.path.join(model_dir, 'Conv1D_regression_SCMneg.h5'))
            model.compile(optimizer='adam', loss='mae', metrics=['mae'])
            scm_neg_batch = model.predict(X_batch, verbose=0)
            del model  # Free memory
        except Exception as e:
            print(f"Error with SCMneg model in batch {batch_idx + 1}: {e}")
            return None, None, None
        
        # Store batch results
        if batch_idx == 0:
            sap_pos_all = sap_pos_batch
            scm_pos_all = scm_pos_batch
            scm_neg_all = scm_neg_batch
        else:
            sap_pos_all = np.vstack([sap_pos_all, sap_pos_batch])
            scm_pos_all = np.vstack([scm_pos_all, scm_pos_batch])
            scm_neg_all = np.vstack([scm_neg_all, scm_neg_batch])
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return sap_pos_all, scm_pos_all, scm_neg_all

# FASTA 파일 생성 함수 최적화
def write_fasta_chunks(sequences, output_path, seq_type='heavy'):
    """
    Write sequences to FASTA file efficiently
    """
    with open(output_path, 'w') as f:
        for seq_dict in sequences:
            for i, name in enumerate(seq_dict['name']):
                seq = seq_dict['heavy'][i] if seq_type == 'heavy' else seq_dict['light'][i]
                f.write(f">{name}\n{seq}\n")

# Main command-line interface function using click
@click.command()
@click.option('--input_csv', required=True, help='Path to the input CSV file.')
@click.option('--output_csv', required=True, help='Path to the output CSV file or directory.')
@click.option('--chunk_size', default=1000, help='Number of sequences to process at once for FASTA generation.')
@click.option('--batch_size', default=5000, help='Batch size for model predictions.')
@click.option('--preprocessing_chunk_size', default=50000, help='Chunk size for reading alignment files.')
def main(input_csv, output_csv, chunk_size, batch_size, preprocessing_chunk_size):
    """
    Predicts DeepViscosity classes for antibody sequences from an input CSV file.
    """
    # 입력 파일명에서 기본 파일명 추출
    input_basename = os.path.splitext(os.path.basename(input_csv))[0]
    
    # 출력 경로 처리 개선
    if os.path.isdir(output_csv):
        # 출력이 디렉토리인 경우, 입력 파일명을 기반으로 출력 파일명 생성
        output_filename = f"{input_basename}_DeepViscosity_predictions.csv"
        output_csv_filepath = os.path.join(output_csv, output_filename)
        output_dir = output_csv
    else:
        # 출력이 파일 경로인 경우
        output_csv_filepath = output_csv
        output_dir = os.path.dirname(output_csv_filepath)
        if not output_dir:
            output_dir = "."

    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Output will be saved to: {output_csv_filepath}")

    # 현재 스크립트의 절대 경로를 얻고, 프로젝트의 루트 디렉토리를 계산합니다.
    # predict.py는 src/deepviscosity/ 안에 있으므로, 두 레벨 위가 루트입니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    print(f"Processing dataset from: {input_csv}")
    
    # Check if FASTA files already exist
    fasta_H_path = os.path.join(output_dir, 'seq_H.fasta')
    fasta_L_path = os.path.join(output_dir, 'seq_L.fasta')
    anarci_output_base = os.path.join(output_dir, 'seq_aligned')
    aligned_csv_H_path = anarci_output_base + '_H.csv'
    aligned_csv_KL_path = anarci_output_base + '_KL.csv'

    if os.path.exists(fasta_H_path) and os.path.exists(fasta_L_path):
        print("Found existing FASTA files. Skipping FASTA generation...")
    else:
        try:
            # 청크 단위로 데이터 처리
            sequence_chunks = process_sequences_in_chunks(input_csv, chunk_size)
            
            # FASTA 파일 생성
            print("Converting sequences to FASTA format...")
            write_fasta_chunks(sequence_chunks, fasta_H_path, 'heavy')
            write_fasta_chunks(sequence_chunks, fasta_L_path, 'light')
        except Exception as e:
            print(f"Error processing input CSV: {e}")
            return

    if os.path.exists(aligned_csv_H_path) and os.path.exists(aligned_csv_KL_path):
        print("Found existing aligned sequence files. Skipping ANARCI alignment...")
    else:
        # Sequence alignment with ANARCI
        print("Performing sequence alignment with ANARCI...")
        
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
    seq_preprocessing(aligned_csv_H_path, aligned_csv_KL_path, combined_seq_txt_path, chunk_size=preprocessing_chunk_size)

    # Load preprocessed sequences
    name_list, seq_list = load_input_data(combined_seq_txt_path)

    # One hot encoding of aligned sequences using batch processing
    X = batch_one_hot_encode(seq_list, batch_size=10000)

    # DeepSP Predictions (models assumed to be in fixed relative paths)
    print("Making DeepSP predictions...")
    # DeepSP 모델의 절대 경로 설정
    deepsp_model_dir = os.path.join(project_root, 'data', 'DeepSP_CNN_model')
    deepsp_descriptors_path = os.path.join(output_dir, 'DeepSP_descriptors.csv') # Path for DeepSP output

    # Use batch processing for large datasets
    sap_pos, scm_pos, scm_neg = process_predictions_in_batches(X, deepsp_model_dir, batch_size=batch_size)
    
    if sap_pos is None or scm_pos is None or scm_neg is None:
        print("Error in batch processing. Exiting.")
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

    # Process ensemble models with memory management
    n_samples = X_scaled.shape[0]
    ensemble_batch_size = min(batch_size, n_samples)  # Use the same batch size
    n_batches = (n_samples + ensemble_batch_size - 1) // ensemble_batch_size
    
    print(f"Processing DeepViscosity ensemble with {n_batches} batches of size {ensemble_batch_size}")
    
    for i in range(102):
        file_name = 'ANN_logo_' + str(i)
        json_model_path = os.path.join(deepviscosity_ann_models_dir, file_name + '.json')
        h5_weights_path = os.path.join(deepviscosity_ann_models_dir, file_name + '.h5')

        try:
            with open(json_model_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(h5_weights_path)
            model.compile(optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
            
            # Process in batches for large datasets
            batch_preds = []
            for batch_idx in range(n_batches):
                start_idx = batch_idx * ensemble_batch_size
                end_idx = min((batch_idx + 1) * ensemble_batch_size, n_samples)
                X_batch = X_scaled[start_idx:end_idx]
                
                pred_batch = model.predict(X_batch, verbose=0)
                batch_preds.append(pred_batch)
            
            # Combine batch predictions
            pred = np.vstack(batch_preds) if len(batch_preds) > 1 else batch_preds[0]
            model_preds.append(pred)
            
            # Free memory
            del model
            import gc
            gc.collect()
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/102 ensemble models")
                
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