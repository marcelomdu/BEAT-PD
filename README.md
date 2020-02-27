# BEAT-PD
BEAT-PD Challenge

Development of a siamese convolutional network for personified classification of accelerometer data from PD patients

For each 20 minute series of each patient, the following preprocessing steps are performed:
- Accelerometer data is first converted to direction change magnitudes (sqrt(dx^2+dy^2+dz^2)-1) and then subdivided in 10 seconds intervals;
- For each interval the Power Spectrum Density is calculated (scipy.signal.welch, window='hann');
- All the PSD graphs (each with 129 points) are stacked, forming a Nx129 array, where N is the number of 10 seconds intervals in the given series;
- For each 20 minute series the patient will have a N x 129 array and the self reported scores for medication use, dyskinesia level and tremor level.

For each patient a siamese convolutional network will be trained on the series arrays, with labels being the self reported scores


