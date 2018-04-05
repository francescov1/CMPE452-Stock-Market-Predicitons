Running the Code

- <i>network.py</i> contains all the required code
- <i>saved_networks</i> folder contains files for previously trained networks. Any network that is saved will be saved here
- <i>5yr-SP-data.csv</i> contains all the required stock price data
- Beneath the function definitions are adjustable parameters and their respective descriptions in comments.
- If <i>load_model</i> parameter is set to True, the program will search in the saved_networks folder for a network with the filename containing the specified number of neurons and epochs parameters.
- If <i>load_model</i> is set to False, the program will also include a plot of losses, and ask the user if they would like to save their newly trained network (upon closing the plots). <b>This will overwrite any previously saved networks with the same number of neurons and epochs.</b>
