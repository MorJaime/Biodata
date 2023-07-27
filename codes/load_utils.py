def time_change(acc_df):
    df=acc_df['timestamp']
    df1=df.astype(np.int64)
    acc_df['timestamp']=df1/1000000
    acc_df['timestamp'] = acc_df['timestamp'].map(lambda x: int(x))
    return acc_df

def setup_dir(path: str):
    """ Setup write directory

    Args.
    ------
    - path: path to write folder

    """
    if not os.path.isdir(path):
        # If selected DIR does not exist, create it.
        os.makedirs(path)
        if os.path.isdir(path):
            logger.info(f"Created Dir: {path}")

    return