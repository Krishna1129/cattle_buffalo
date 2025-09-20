# cow-buffalo-classifier

## Project Overview: 
This project focuses on classifying images of cattle into two primary categories: cows and buffaloes. After the initial classification, the model further identifies the breed of the animal, such as Holstein or Murrah. This system aids farmers, veterinarians, and livestock specialists by providing an efficient and accurate way to identify both the type and breed of livestock, enabling better management and care.

## Project Details: 
The model is built using Python and PyTorch for deep learning, with the web interface powered by Streamlit. Users can easily upload images through the interface to receive predictions on the animal's type (cow or buffalo) and its breed. The Cow/Buffalo Classification model was trained for 50 epochs on an RTX 3060 GPU, achieving a training accuracy of 98% and a test accuracy of 97%. The Breed Classification model, trained for 100 epochs on the same GPU, reached a training accuracy of 94% and a test accuracy of 89%. Version 1 of the application currently supports basic cow and buffalo classification, with plans for future updates to include additional animal types and breeds, further enhancing the system's capabilities.

## Tech Stack: 
Python, PyTorch, pillow, Streamlit, Flask
