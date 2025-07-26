from simuq.provider import BaseProvider
from simuq.solver import generate_as
import numpy as np
import json
import re
from neal import SimulatedAnnealingSampler

from simuq.aais import ising
from simuq.dwave.dwave_transpiler import DwaveTranspiler


class DWaveSimProvider(BaseProvider):
    def __init__(self, **sampler_kwargs):
        # Initialize without API key requirements
        super().__init__()
        self._samples = None
        # Store all kwargs for sampling methods
        self.sampling_args = sampler_kwargs

    def compile(self,
                qs,
                anneal_schedule=[[0, 0], [20, 1]],
                chain_strength=0.5,
                verbose=0):
        h = [0 for _ in range(qs.num_sites)]
        J = {}
        for ham in qs.evos[0][0].ham:
            keys = list(ham[0].d.keys())
            vals = list(ham[0].d.values())
            if 'Z' in vals:
                if len(vals) == 1:
                    h[keys[0]] = ham[1]
                elif len(vals) == 2 and ham[1] != 0:
                    J[(keys[0], keys[1])] = ham[1]
        self.prog = h, J, anneal_schedule
        self.chain_strength = chain_strength
        return h, J

    def compare_qubo(self, q1, q2):
        numDifferent = 0
        if len(q1.keys()) != len(q2.keys()): return False
        for (d1, d2) in q1.keys():
            if abs(q1[(d1, d2)] - q2[(int(d1), int(d2))]) > 10**-3:
                numDifferent += 1
        return numDifferent

    def run(self, shots=100, solver=None):
        self.shots = shots
        if self.prog is None:
            raise Exception("No compiled job in record.")
        
        # Use SimulatedAnnealingSampler instead of DWaveSampler
        sampler = SimulatedAnnealingSampler()
        h, J, anneal_schedule = self.prog
        
        # Convert h to dictionary format expected by neal with proper dtype
        h_dict = {i: float(h[i]) for i in range(len(h)) if h[i] != 0}
        
        # Ensure J values are proper float64 and non-zero
        J_dict = {k: float(v) for k, v in J.items() if v != 0}
        
        # Combine num_reads with other sampling arguments
        sampling_kwargs = {'num_reads': self.shots}
        sampling_kwargs.update(self.sampling_args)
        
        response = sampler.sample_ising(h_dict, J_dict, **sampling_kwargs)
        
        self.samples = list(response.samples())
        
        # Simulated annealing doesn't have QPU timing, so setting these to 0
        self.time_on_machine = 0.0
        self.avg_qpu_time = 0.0
        self.num_occurrences = list(response.data_vectors['num_occurrences'])
        return response

    def isingToqubo(self, h, J):
        # edited for verstatility
        # Handle both dict and list inputs for h
        if isinstance(h, dict):
            h_max = max(h.keys()) if h else -1
            j_max = max([max(k) for k in J.keys()]) if J else -1
            max_node = max(h_max, j_max)
            n = max_node + 1 if max_node >= 0 else 0
        else:
            n = len(h)
        
        QUBO = {}

        for i in range(n):
            s = 0
            for ii in range(n):
                if (i,ii) in J:
                    s += J[(i,ii)]
                if (ii,i) in J:
                    s += J[(ii,i)]

            h_val = h.get(i, 0) if isinstance(h, dict) else (h[i] if i < len(h) else 0)
            diagonal_val = -2 * (h_val + s)
            if abs(diagonal_val) > 1e-12:  # Only include non-zero terms
                QUBO[(i,i)] = diagonal_val

            for j in range(i+1, n):
                if (i,j) in J and abs(J[(i,j)]) > 1e-12:
                    QUBO[(i,j)] = 4 * J[(i,j)]

        return QUBO

    def run_qubo(self):
        if self.prog is None:
            raise Exception("No compiled job in record.")
        
        # Use SimulatedAnnealingSampler instead of DWaveSampler
        sampler = SimulatedAnnealingSampler()
        h, J, anneal_schedule = self.prog
        h_dict = {i: float(h[i]) for i in range(len(h)) if h[i] != 0}
        J_dict = {k: float(v) for k, v in J.items() if v != 0}
        qubo = self.isingToqubo(h_dict, J_dict)
        
        # Ensure QUBO values are also proper float64
        qubo_clean = {k: float(v) for k, v in qubo.items() if v != 0}
        
        # Combine num_reads with other sampling arguments
        sampling_kwargs = {'num_reads': self.shots}
        sampling_kwargs.update(self.sampling_args)
        
        response = sampler.sample_qubo(qubo_clean, **sampling_kwargs)
        self.samples = list(response.samples())

    def results(self):
        if self.samples == None:
            raise Exception("Job has not been run yet.")
        return self.samples 