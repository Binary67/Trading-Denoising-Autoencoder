import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

class Conv1DAutoencoderDenoising:
    """
    A Convolutional Denoising Autoencoder (CAE) for trading data denoising.
    Implements data loading, transformation, model building, training, evaluation, 
    saving/loading, and prediction functionalities.
    """
    
    def __init__(self):
        # Set device to GPU if available, otherwise CPU
        self.Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NoisyData = None
        self.CleanData = None
        self.ScalerParameters = None  # To store mean and std for scaling
        self.Model = None

    def LoadData(self, NoisyData, CleanData):
        """
        Load noisy and clean trading data from specified file paths.
        
        Args:
            NoisyDataPath (str): File path for the noisy trading data.
            CleanDataPath (str): File path for the clean trading data.
        """
        try:
            self.NoisyData = NoisyData.copy()
            self.CleanData = CleanData.copy()
        except Exception as Error:
            print(f"Error loading data: {Error}")
            raise Error

        # Validate required columns exist in both dataframes
        RequiredColumns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        for Column in RequiredColumns:
            if Column not in self.NoisyData.columns or Column not in self.CleanData.columns:
                raise ValueError(f"Missing required column: {Column}")
        
        # Validate that the Datetime column aligns between the two datasets
        if not self.NoisyData['Datetime'].equals(self.CleanData['Datetime']):
            raise ValueError("Datetime columns do not match between noisy and clean data.")

    def TransformData(self, TradingData, FeaturesToUse=['Open', 'High', 'Low', 'Close', 'Volume'], 
                      SequenceLength=64, Stride=1):
        """
        Transform the trading data into sequences suitable for a 1D Convolutional Autoencoder.
        This includes feature selection, scaling, and sequence creation.
        
        Args:
            TradingData (pd.DataFrame): The trading data to transform.
            FeaturesToUse (list of str): List of feature column names to use.
            SequenceLength (int): The length of the time series sequences.
            Stride (int): The step size for the sliding window.
        
        Returns:
            torch.Tensor: A tensor of shape (number_of_sequences, SequenceLength, num_features)
                          representing the transformed data.
        """
        # Select specified features and convert to numpy array
        DataArray = TradingData[FeaturesToUse].copy().values.astype(np.float32)
        
        # Scaling: Standardization (Z-score)
        # If scaling parameters have not been computed yet, calculate and store them.
        if self.ScalerParameters is None:
            MeanArray = np.mean(DataArray, axis=0)
            StdArray = np.std(DataArray, axis=0)
            self.ScalerParameters = {'Mean': MeanArray, 'Std': StdArray}
        else:
            MeanArray = self.ScalerParameters['Mean']
            StdArray = self.ScalerParameters['Std']
        
        # Apply scaling
        DataScaled = (DataArray - MeanArray) / (StdArray + 1e-8)
        
        # Create sequences using a sliding window approach
        NumSamples = DataScaled.shape[0]
        SequenceList = []
        for StartIndex in range(0, NumSamples - SequenceLength + 1, Stride):
            Sequence = DataScaled[StartIndex:StartIndex + SequenceLength]
            SequenceList.append(Sequence)
        SequencesArray = np.array(SequenceList)
        
        # Convert to PyTorch tensor
        TransformedData = torch.tensor(SequencesArray, dtype=torch.float32)
        return TransformedData

    def BuildModel(self, SequenceLength, NumFeatures, LatentDim=32, Filters=[32, 64, 128], 
                   KernelSize=3, Stride=1):
        """
        Build and initialize the 1D Convolutional Autoencoder model.
        
        Args:
            SequenceLength (int): The length of the input sequences.
            NumFeatures (int): The number of features per time step.
            LatentDim (int): The dimension of the latent space.
            Filters (list of int): Filter sizes for each convolutional layer.
            KernelSize (int): Kernel size for convolutional layers.
            Stride (int): Stride for convolutional layers.
        """
        class CAEModel(nn.Module):
            def __init__(self, SequenceLength, NumFeatures, LatentDim, Filters, KernelSize, Stride):
                super(CAEModel, self).__init__()
                # Encoder: series of Conv1d, ReLU, and MaxPool1d layers
                self.Encoder = nn.Sequential(
                    nn.Conv1d(NumFeatures, Filters[0], KernelSize, stride=Stride, padding=KernelSize//2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(Filters[0], Filters[1], KernelSize, stride=Stride, padding=KernelSize//2),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Conv1d(Filters[1], Filters[2], KernelSize, stride=Stride, padding=KernelSize//2),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
                # Calculate the length after pooling (assumes SequenceLength is divisible by 8)
                PooledLength = SequenceLength // 2 // 2 // 2  # SequenceLength divided by 8
                self.FlattenSize = Filters[2] * PooledLength
                self.EncoderFc = nn.Linear(self.FlattenSize, LatentDim)
                
                # Decoder: Fully connected layer followed by ConvTranspose1d and upsampling layers
                self.DecoderFc = nn.Linear(LatentDim, self.FlattenSize)
                self.Decoder = nn.Sequential(
                    nn.ConvTranspose1d(Filters[2], Filters[1], KernelSize, stride=Stride, padding=KernelSize//2),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.ConvTranspose1d(Filters[1], Filters[0], KernelSize, stride=Stride, padding=KernelSize//2),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2),
                    nn.ConvTranspose1d(Filters[0], NumFeatures, KernelSize, stride=Stride, padding=KernelSize//2),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2)
                )
            
            def forward(self, InputX):
                # InputX shape: (Batch, SequenceLength, NumFeatures)
                # Permute to (Batch, NumFeatures, SequenceLength) for Conv1d
                InputX = InputX.permute(0, 2, 1)
                Encoded = self.Encoder(InputX)
                BatchSize = Encoded.size(0)
                EncodedFlat = Encoded.view(BatchSize, -1)
                Latent = self.EncoderFc(EncodedFlat)
                
                # Decoder
                DecodedFlat = self.DecoderFc(Latent)
                # Reshape to (Batch, Filters[2], PooledLength)
                PooledLength = self.FlattenSize // Filters[2]
                DecodedReshaped = DecodedFlat.view(BatchSize, Filters[2], PooledLength)
                Reconstructed = self.Decoder(DecodedReshaped)
                # Permute back to (Batch, SequenceLength, NumFeatures)
                Reconstructed = Reconstructed.permute(0, 2, 1)
                return Reconstructed

        # Initialize the model and move it to the appropriate device
        self.Model = CAEModel(SequenceLength, NumFeatures, LatentDim, Filters, KernelSize, Stride)
        self.Model.to(self.Device)

    def TrainModel(self, TrainNoisyData, TrainCleanData, ValNoisyData, ValCleanData, 
                   Epochs=100, BatchSize=512, LearningRate=0.001, OptimizerName='Adam', 
                   LossFunctionName='MSE', SaveBestModel=True, ModelSavePath='ModelsPath/Autoencoder.pth'):
        """
        Train the CAE model using the provided training and validation data.
        
        Args:
            TrainNoisyData (torch.Tensor): Transformed noisy training data.
            TrainCleanData (torch.Tensor): Transformed clean training data.
            ValNoisyData (torch.Tensor): Transformed noisy validation data.
            ValCleanData (torch.Tensor): Transformed clean validation data.
            Epochs (int): Number of training epochs.
            BatchSize (int): Batch size for training.
            LearningRate (float): Learning rate for the optimizer.
            OptimizerName (str): Optimizer to use ('Adam' or 'RMSprop').
            LossFunctionName (str): Loss function to use ('MSE' or 'MAE').
            SaveBestModel (bool): Whether to save the best model based on validation loss.
            ModelSavePath (str): File path to save the best model.
        """
        # Create DataLoaders for training and validation
        TrainDataset = TensorDataset(TrainNoisyData, TrainCleanData)
        ValDataset = TensorDataset(ValNoisyData, ValCleanData)
        TrainLoader = DataLoader(TrainDataset, batch_size=BatchSize, shuffle=True)
        ValLoader = DataLoader(ValDataset, batch_size=BatchSize, shuffle=False)
        
        # Select optimizer
        if OptimizerName.lower() == 'adam':
            Optimizer = optim.Adam(self.Model.parameters(), lr=LearningRate)
        elif OptimizerName.lower() == 'rmsprop':
            Optimizer = optim.RMSprop(self.Model.parameters(), lr=LearningRate)
        else:
            raise ValueError("Unsupported optimizer. Choose 'Adam' or 'RMSprop'.")
        
        # Select loss function
        if LossFunctionName.lower() == 'mse':
            LossFunction = nn.MSELoss()
        elif LossFunctionName.lower() == 'mae':
            LossFunction = nn.L1Loss()
        else:
            raise ValueError("Unsupported loss function. Choose 'MSE' or 'MAE'.")
        
        BestValLoss = np.inf
        for Epoch in range(1, Epochs + 1):
            # Training Phase
            self.Model.train()
            TrainLoss = 0.0
            for NoisyBatch, CleanBatch in TrainLoader:
                NoisyBatch = NoisyBatch.to(self.Device)
                CleanBatch = CleanBatch.to(self.Device)
                Optimizer.zero_grad()
                Outputs = self.Model(NoisyBatch)
                LossValue = LossFunction(Outputs, CleanBatch)
                LossValue.backward()
                Optimizer.step()
                TrainLoss += LossValue.item() * NoisyBatch.size(0)
            TrainLoss /= len(TrainLoader.dataset)
            
            # Validation Phase
            self.Model.eval()
            ValLoss = 0.0
            with torch.no_grad():
                for NoisyBatch, CleanBatch in ValLoader:
                    NoisyBatch = NoisyBatch.to(self.Device)
                    CleanBatch = CleanBatch.to(self.Device)
                    Outputs = self.Model(NoisyBatch)
                    LossValue = LossFunction(Outputs, CleanBatch)
                    ValLoss += LossValue.item() * NoisyBatch.size(0)
            ValLoss /= len(ValLoader.dataset)
            
            print(f"Epoch {Epoch}/{Epochs}, Train Loss: {TrainLoss:.6f}, Val Loss: {ValLoss:.6f}")
            
            # Save best model based on validation loss
            if SaveBestModel and ValLoss < BestValLoss:
                BestValLoss = ValLoss
                torch.save(self.Model.state_dict(), ModelSavePath)
        
        print("Training complete.")

    def SaveModel(self, SavePath):
        """
        Save the trained CAE model's state dictionary to a file.
        
        Args:
            SavePath (str): File path to save the model state.
        """
        torch.save(self.Model.state_dict(), SavePath)
        print(f"Model saved to {SavePath}.")

    def LoadModel(self, LoadPath, ResumeTraining=False):
        """
        Load a pre-trained CAE model's state dictionary from a file.
        
        Args:
            LoadPath (str): File path containing the saved model state.
            ResumeTraining (bool): If True, set the model to training mode.
        """
        StateDict = torch.load(LoadPath, map_location=self.Device)
        self.Model.load_state_dict(StateDict)
        self.Model.to(self.Device)
        if ResumeTraining:
            self.Model.train()
        else:
            self.Model.eval()
        print(f"Model loaded from {LoadPath}.")

    def EvaluateModel(self, ValNoisyData, ValCleanData, BatchSize=32):
        """
        Evaluate the CAE model on validation data.
        
        Args:
            ValNoisyData (torch.Tensor): Transformed noisy validation data.
            ValCleanData (torch.Tensor): Transformed clean validation data.
            BatchSize (int): Batch size for evaluation.
        
        Returns:
            dict: Dictionary containing evaluation metrics (e.g., ValidationLoss, RMSE).
        """
        LossFunction = nn.MSELoss()
        ValDataset = TensorDataset(ValNoisyData, ValCleanData)
        ValLoader = DataLoader(ValDataset, batch_size=BatchSize, shuffle=False)
        
        self.Model.eval()
        TotalLoss = 0.0
        with torch.no_grad():
            for NoisyBatch, CleanBatch in ValLoader:
                NoisyBatch = NoisyBatch.to(self.Device)
                CleanBatch = CleanBatch.to(self.Device)
                Outputs = self.Model(NoisyBatch)
                LossValue = LossFunction(Outputs, CleanBatch)
                TotalLoss += LossValue.item() * NoisyBatch.size(0)
        AvgLoss = TotalLoss / len(ValLoader.dataset)
        RMSE = np.sqrt(AvgLoss)
        Metrics = {'ValidationLoss': AvgLoss, 'RMSE': RMSE}
        print(f"Evaluation Metrics: {Metrics}")
        return Metrics

    def Predict(self, NoisyDataForPrediction):
        """
        Denoise new, unseen noisy trading data using the trained CAE model and return a 
        DataFrame with columns similar to the trading data OHLCV format.
        
        Args:
            NoisyDataForPrediction (pd.DataFrame or torch.Tensor): 
                The noisy trading data to denoise. If a DataFrame is provided, it should include 
                the "Datetime" column along with OHLCV data. The DataFrame will be transformed.
        
        Returns:
            pd.DataFrame: A DataFrame with denoised OHLCV data (and "Datetime" column if available).
        """
        # If input is a DataFrame, store a copy to retain the original "Datetime" column.
        if isinstance(NoisyDataForPrediction, pd.DataFrame):
            OriginalDataFrame = NoisyDataForPrediction.copy()
            TensorData = self.TransformData(NoisyDataForPrediction)
        elif torch.is_tensor(NoisyDataForPrediction):
            TensorData = NoisyDataForPrediction
            OriginalDataFrame = None
        else:
            raise TypeError("Input data must be a pandas DataFrame or a torch.Tensor.")
        
        TensorData = TensorData.to(self.Device)
        self.Model.eval()
        with torch.no_grad():
            DenoisedOutput = self.Model(TensorData)
        
        # Inverse transform the scaled data if scaling was applied.
        if self.ScalerParameters is not None:
            DenoisedNp = DenoisedOutput.cpu().numpy()  # shape: (num_sequences, sequence_length, num_features)
            StdArray = self.ScalerParameters['Std']
            MeanArray = self.ScalerParameters['Mean']
            DenoisedNp = DenoisedNp * StdArray + MeanArray
        else:
            DenoisedNp = DenoisedOutput.cpu().numpy()
        
        # If the original input was a DataFrame, aggregate the overlapping sequences.
        if OriginalDataFrame is not None:
            N = OriginalDataFrame.shape[0]  # original number of rows
            L = TensorData.shape[1]         # sequence length used in transformation
            S = DenoisedNp.shape[0]         # number of sequences = N - L + 1
            num_features = DenoisedNp.shape[2]
            
            # Initialize accumulators for summing predictions and counting overlaps.
            Accumulator = np.zeros((N, num_features))
            Count = np.zeros(N)
            
            # For each sequence, add its predictions to the corresponding rows.
            for i in range(S):
                Accumulator[i:i+L] += DenoisedNp[i]
                Count[i:i+L] += 1
            
            # Compute the average prediction for each row.
            AveragedOutput = Accumulator / Count[:, None]
            
            # Create a DataFrame with the standard OHLCV columns.
            FeatureColumns = ['Open', 'High', 'Low', 'Close', 'Volume']
            ResultDf = pd.DataFrame(AveragedOutput, columns=FeatureColumns)
            
            # If the original DataFrame contains "Datetime", add it as the first column.
            if 'Datetime' in OriginalDataFrame.columns:
                ResultDf.insert(0, 'Datetime', OriginalDataFrame['Datetime'].values)
            return ResultDf
        else:
            # If the input was a tensor (i.e. no original DataFrame), 
            # return a DataFrame with a flattened multi-index structure.
            S, L, F = DenoisedNp.shape
            Flattened = DenoisedNp.reshape(S * L, F)
            FeatureColumns = ['Open', 'High', 'Low', 'Close', 'Volume']
            ResultDf = pd.DataFrame(Flattened, columns=FeatureColumns)
            return ResultDf
