class Encoder(object):

    def encode_X(self, df):
        try:
            df_without_label = df.drop(labels=['label'], axis=1)
        except ValueError:
            df_without_label = df

        X = df_without_label.values
        
        # normalize the values between 0.0 and 1.0
        X = X / 255.0
        
        return X

    def encode_y(self, df):
        return df['label'].values
