"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function
import logging
from optimizer import Optimizer
from tqdm import tqdm
import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename='log.txt'
)

def train_networks(networks, dataset, type_ser):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset, type_ser)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks.

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.

    """
    total_accuracy = 0
    for network in networks:
        if network.accuracy != -1 and network.accuracy != None:
            total_accuracy += network.accuracy
        else:
            continue

    return total_accuracy / len(networks)

def get_min_mse(networks):
    """Get the average accuracy for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        float: The average accuracy of a population of networks/genomes.

    """
    min_accuracy = 1000000
    min_network  = None
    for network in networks:
        if network.accuracy != -1 and network.accuracy < min_accuracy:
            min_accuracy = network.accuracy
            min_network=network
        else:
            continue
    return min_accuracy,min_network

def generate(generations, population, k, nn_param_choices, dataset, type_ser):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
    logging.info("***generate(generations, population, nn_param_choices, dataset)***")
    optimizer = Optimizer(nn_param_choices)
    #print("--1--")
    networks = optimizer.create_population(population)
    #print("--2--")
    min_networks=[]
    # Evolve the generation.
    for i in range(generations):
        logging.info("*** Now in generation %d of %d ***" %(i + 1, generations))
        #print_networks(networks)
        # Train and get accuracy for networks.
        train_networks(networks, dataset, type_ser)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        min_mse,net =get_min_mse(networks)
        #print_networks(net)
        #net.print_network()
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f" % (average_accuracy ))#* 100
        logging.info('-'*80) #-----------
        logging.info("Generation min_mse: %.2f" % (min_mse ))#* 100
        logging.info('-'*80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
        min_networks.append(net)
    # Sort our final population.
    # Sort our final population according to performance.
    networks=[x for x in networks if x.accuracy !=-1 and x.accuracy != None]
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=False)
    print("Generation done")

    # Print out the top 5 networks.
    logging.info("top 5 networks Generation" )
    logging.info('='*80) #-----------
    print_networks(networks[:5])
    
    pd.DataFrame(networks[:5]).to_csv(file_network, sep=',', encoding='utf-8', mode='a', header=True)
    for n in range(0,5):
        score_table.loc[n+k,2:] = networks[n].param, networks[n].param_seasonal, networks[n].accuracy
    
    logging.info("min networks Generation" )
    logging.info('-'*80) #-----------
    min_networks = sorted(min_networks, key=lambda x: x.accuracy, reverse=False)
    print_networks(min_networks)
    return networks
    
def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks

    """
    logging.info('#'*80)
    for network in networks:
        network.print_network()

def main():
    """Evolve a network."""
    generations  = [20]#, 10, 20, 50, 100, 200]  # Number of times to evole the population.
    populations  = [10, 50, 100, 200, 300, 400, 500]  # Number of networks in each generation.
    dataset = 'cifar10'
    k=0
    nn_param_choices = {
        'p_values' : range(0, 12),#[0, 1, 4, 6, 8, 10],
        'd_values' : range(0, 2),#[0, 1, 2], #range(0, 3)
        'q_values' : range(0, 12),#[0, 1, 2], #range(0, 3)
        'sp_values': range(0, 12),#[0, 1, 4, 6, 8, 10],
        'sd_values': range(0, 2),#[0, 1, 2],
        'sq_values': range(0, 12) #[0, 1, 2]
    }
    type_ser ='normalized'
    #for type_ser in ['normal', 'log', 'loglog']: #'normalized',
    for generation in generations: #'normalized',
        for population in populations:
            print("$$$$$$$$$$$$$$$$$$$$ %s $$$$$$$$$$$$$$$$$"% (type_ser))
            logging.info("***Evolving %d generations with population %d***" % (generation, population))
            networks = generate(generation, population, k, nn_param_choices, dataset, type_ser)
            k=k+5
            clear_output()
    return networks
if __name__ == '__main__':
    networks = main()
