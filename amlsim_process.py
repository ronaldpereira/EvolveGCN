import argparse

import pandas as pd

SEED = 1212


def load_transactions(file_path: str) -> pd.DataFrame:
    dtypes = {
        'tran_id': int,
        'orig_acct': int,
        'bene_acct': int,
        'tx_type': str,
        'base_amt': float,
        'tran_timestamp': str,
        'is_sar': int,
        'alert_id': int
    }

    df = pd.read_csv(file_path, dtype=dtypes, parse_dates=['tran_timestamp'])

    return df


def generate_feature_vectors(input_file_path: str, output_filepath: str):
    feat_df = load_transactions(input_file_path)
    feat_df.drop(['tran_id', 'tx_type', 'alert_id'], axis=1, inplace=True)

    feat_df['tran_timestamp'] = (feat_df['tran_timestamp'] -
                                 pd.Timestamp('1970-01-01 00:00:00+00:00')) // pd.Timedelta('1s')

    feat_df.to_csv('%s' % output_filepath, index=False, header=False)


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='AMLSim transactions data input file path.')
    parser.add_argument('output_filepath', type=str, help='Output dataset filename.')

    args = parser.parse_args()

    return args


def generate_amlsim_data():
    args = arg_parser()
    generate_feature_vectors(args.input, args.output_filepath)


if __name__ == '__main__':
    generate_amlsim_data()
