import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
import pickle
import time
import sys
import pandas as pd
import csv
import gzip

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
class OurFed(fl.server.strategy.FedAvg):
    def __init__(
            self,
            fraction_fit: float = 0.1,
            fraction_eval: float = 0.1,
            min_fit_clients: int = 2,
            min_eval_clients: int = 2,
            min_available_clients: int = 2,
            eval_fn = None,
            on_fit_config_fn = None,
            on_evaluate_config_fn = None,
            accept_failures: bool = True,
            initial_parameters = None,
    ) -> None:
        super(OurFed, self).__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters
        )
        self.kb = []
        self.totTimeStart = 0.0
        self.totTimeEnd = 0.0
        self.serverCompTime = []
        self.totalTime = []
        self.totalCommTime = []
        self.avgOneWayCommTime = []
        self.clientCompTime = []
        self.serverToClientCommSize = []
        self.clientToServerCommSize = []
        self.training_acc = []
        self.test_acc = []
        myheaders = ['round','total-time','server-time','client-time','total-comm-time','avg-one-way-comm-time','server-to-client-comm-size','client-to-server-comm-size','training-acc','test-acc']
        self.filename = 'experiment.csv'
        with open(self.filename, 'w', newline='') as myfile:
            writer = csv.writer(myfile)
            writer.writerow(myheaders)
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager
    ) :
        """Configure the next round of training."""
        start = time.time()
        self.totTimeStart = start
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        config['round'] = rnd


        print("round is now",rnd)
        kb_converted = []
        kb_converted_string = ""
        if(len(self.kb) > 0):
            for tensor in range(len(self.kb[0])):
                internal = []
                for client in range(len(self.kb)):
                    internal.append(self.kb[client][tensor])
                kb_converted.append(np.stack(internal,axis=-1))
            kb_converted_string = pickle.dumps(kb_converted)
            kb_converted_string = gzip.compress(kb_converted_string)
            self.kb = []
        config['kb'] = kb_converted_string

        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        end = time.time()
        self.serverCompTime.append(end - start)
        paramSize = sum([len(x) for x in parameters.tensors])
        configSize = sys.getsizeof(config['kb'])
        self.serverToClientCommSize.append(paramSize + configSize)
        # print("Server side compute time prior to sending parameters:",end - start)
        # print("size of parameters being sent to client:", paramSize)
        # print("size of knowledge base sent to the client:", configSize)
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        config['round'] = rnd
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        start = time.time()
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}



        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = weights_to_parameters(aggregate(weights_results))
        # print("--the length of data is---- ")
        kb = []
        avg_client_exec_time = 0.0
        avg_client_parameter_size = 0.0
        avg_client_config_size = 0.0
        avg_client_training_accuracy = 0.0
        num_clients = 0
        for _,fitres in results:
            if (fitres.metrics['kb'] != ""):
                kb.append(pickle.loads(gzip.decompress(fitres.metrics['kb'])))
            avg_client_parameter_size += fitres.metrics['parameter_size']
            avg_client_config_size += fitres.metrics['kb_size']
            avg_client_exec_time += fitres.metrics['clientExecTime']
            avg_client_training_accuracy += fitres.metrics['train_acc']
            num_clients += 1

        self.kb = kb
        # Aggregate custom metrics if aggregation fn was provided
        end = time.time()
        self.totTimeEnd = end
        self.totalTime.append(self.totTimeEnd - self.totTimeStart)
        self.serverCompTime[-1] += end - start #add the overhead from the aggregation phase too
        self.clientCompTime.append(avg_client_exec_time/num_clients)
        self.totalCommTime.append(self.totalTime[-1] - self.serverCompTime[-1] - self.clientCompTime[-1])
        self.avgOneWayCommTime.append(self.totalCommTime[-1]/2)
        self.clientToServerCommSize.append(avg_client_parameter_size/num_clients + avg_client_config_size/num_clients)
        self.training_acc.append(avg_client_training_accuracy/num_clients)
        # print("server side execution time:", self.serverCompTime[-1])
        # print("average client-side execution time:", avg_client_exec_time/num_clients)
        # print("average parameter size of clients:", avg_client_parameter_size/num_clients)
        # print("average knowledge base size of clients:", avg_client_config_size/num_clients)
        # print("total execution time for round:", self.totTimeEnd - self.totTimeStart)
        # print("total communication time between the clients and server:", self.totalCommTime[-1])
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        avg_accuracy = 0.0
        tot_num_examples = 0
        for _, evaluate_res in results:
            avg_accuracy += evaluate_res.num_examples * evaluate_res.metrics['accuracy']
            tot_num_examples += evaluate_res.num_examples
        print("Avg accuracy of clients after round "+str(rnd)+" is: {:.2f}".format(avg_accuracy/tot_num_examples))
        self.test_acc.append(avg_accuracy/tot_num_examples)
            #avg_accuracy += evaluate_res.num_examples * evaluate_res.accuracy
        self.write_data_to_file(rnd)
        return loss_aggregated, {}

    def write_data_to_file(self, rnd):
        row = []
        if(rnd != -1):
            row.append(rnd)
            row.append(self.totalTime[-1])
            row.append(self.serverCompTime[-1])
            row.append(self.clientCompTime[-1])
            row.append(self.totalCommTime[-1])
            row.append(self.avgOneWayCommTime[-1])
            row.append(self.serverToClientCommSize[-1])
            row.append(self.clientToServerCommSize[-1])
            row.append(self.training_acc[-1])
            row.append(self.test_acc[-1])
            with open(self.filename, 'a', newline='') as myfile:
                writer = csv.writer(myfile)
                writer.writerow(row)
        else:
            pass
