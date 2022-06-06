'''
created by Imdadul Haque
Date: 06/12/2022
'''

import os
import logging
import pytest
import churn_library as cls


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)


@pytest.fixture(name="data_f", scope="session")
def data_fixture():
    '''
    returns the original dataframe
    '''
    try:
        data_frame = cls.import_data("./data/bank_data.csv")
        logging.info("Fixture import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Fixture import_eda: The file wasn't found")
        raise err
    return data_frame


def test_import(data_f):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    # try:
    # 	df = import_data("./data/bank_data.csv")
    # 	logging.info("Testing import_data: SUCCESS")
    # except FileNotFoundError as err:
    # 	logging.error("Testing import_eda: The file wasn't found")
    # 	raise err

    try:
        assert data_f.shape[0] > 0
        assert data_f.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(data_f):
    '''
    test perform eda function
    '''
    cls.perform_eda(data_f)
    try:
        assert os.path.isdir('./images/')
        logging.info("Testing perform_eda: Images folder exists")
    except AssertionError as err:
        logging.error("Testing perform_eda: Images folder wasn't found")
        raise err

    plot_lst = [
        'churn_hist.png',
        'customer_age.png',
        'marital_status.png',
        'total_trans_ct.png',
        'correlation_heatmap.png']

    for plot in plot_lst:
        try:
            assert os.path.isfile('./images/' + plot)
        except AssertionError as err:
            logging.error("Testing perform_eda: %s was not found", plot)
            raise err
    logging.info("Testing perform_eda: SUCCESS")


@pytest.fixture(name="encoder_helper_f", scope="session")
def encoder_helper_fixture(data_f):
    '''
    fixture for the encoder helper function
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    encoded_df = cls.encoder_helper(data_f, cat_columns)
    logging.info("Fixture encoder_helper: SUCCESS")
    return encoded_df, cat_columns


def test_encoder_helper(encoder_helper_f):
    '''
    test encoder helper
    '''
    encoded_df, cat_columns = encoder_helper_f

    for cat in cat_columns:
        try:
            assert cat + '_Churn' in encoded_df.columns
        except AssertionError as err:
            logging.error(
                "Testing encoder_helper: %s_churn column is missing", cat)
            raise err
    logging.info("Testing encoder_helper: SUCCESS")


@pytest.fixture(name="perform_feature_engineering_f", scope="session")
def perform_feature_engineering_fixture(encoder_helper_f):
    '''
    fixture for the perform feature engineering
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    encoded_df, _ = encoder_helper_f
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_df)
    logging.info("Fixture perform_feature_engineering: SUCCESS")
    return x_train, x_test, y_train, y_test, keep_cols


def test_perform_feature_engineering(perform_feature_engineering_f):
    '''
    test perform_feature_engineering
    '''

    x_train, x_test, y_train, y_test, keep_cols = perform_feature_engineering_f

    try:
        for col in keep_cols:
            assert col in x_train.columns
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: %s not in column list", col)
        raise err

    try:
        assert x_train.shape[0] == y_train.shape[0]
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: X and y training size doesn't match")
        raise err

    try:
        assert x_test.shape[0] == y_test.shape[0]
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: X and y test size doesn't match")
        raise err

    logging.info("Testing perform_feature_engineering: SUCCESS")


def test_train_models(perform_feature_engineering_f):
    '''
    test train_models
    '''
    x_train, x_test, y_train, y_test, _ = perform_feature_engineering_f
    cls.train_models(x_train, x_test, y_train, y_test)

    try:
        assert os.path.isfile('./models/rfc_model.pkl')
        logging.info("Testing train_models: Random Forest model exists")
    except AssertionError as err:
        logging.error("Testing train_models: Random Forest model not found")
        raise err

    try:
        assert os.path.isfile('./models/lr_model.pkl')
        logging.info("Testing train_models: Logistic Regression model exists")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Logistic Regression model not found")
        raise err


# if __name__ == "__main__":
    # test_import(cls.import_data)
    # test_eda(cls.perform_eda)
    # test_encoder_helper(cls.encoder_helper)
    # test_perform_feature_engineering(cls.perform_feature_engineering)
    # test_train_models(cls.train_models)
