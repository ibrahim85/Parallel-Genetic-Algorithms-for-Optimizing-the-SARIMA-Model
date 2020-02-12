# Parallel Genetic Algorithms for Optimizing the SARIMA Model

This is an example of how we can use a genetic algorithm in an attempt to find the optimal SARIMA parameters for prediction tasks. 

On the NCDC dataset, we are able to quickly find the best SARIMA model. 

## the basic idea of this model is:

![GA_ARIMA_Parallel2](GA_ARIMA_Parallel2.pdf)


## To run

To run the brute force algorithm:

![Final_code](Final_code.ipynb)

```Final_code.ipynb```

To run the brute force algorithm:

```python3 brute.py```

To run the genetic algorithm:

```paper6--GA--SARIMAX and arima.ipynb```

 To plot the forecasting and other figures:
 
 use this file
 
```Serial SARIMA-plots model.ipynb```

# How to use Google Colab to run the code

https://www.geeksforgeeks.org/how-to-use-google-colab/

You can set your SARIMA parameter choices by editing each of those files first. You can also choose whether to use the NCDC or other datasets. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/ibrahim85/Genetic-Alg-and-SARIMA/blob/master/Genetic%20Alg%20and%20SARIMA/Serial%20SARIMA-%20plots%20model.ipynb)

For more, see this blog post: https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

For a more robust implementation that you can use in your projects, take a look at Jan Liphardt's implementation, DeepEvolve.

## License

MIT


