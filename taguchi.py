# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: TFM
#     language: python
#     name: tfm
# ---

# %%
from typing import List
import numpy as np

class Variable: 
    _name = ""
    _values = []
    def __init__(self, name:str, values: List): 
        self._name = name
        self._values = values
    def __str__(self): 
        return f"Variable(name = '{self._name}', values = {self._values})"

    @property
    def values(self) -> List: 
        return self._values
    
    @property
    def name(self) -> str: 
        return self._name
        
    @property
    def n_values(self) -> int: 
        return len(self._values)


class TaguchiOpt: 
    _n_exp = 0
    _variables: List[Variable] = []
    _Q = 0
    _N = 0
    _J = 0 
    _M = 0 
    _OA = None
    _OA_values = None

    @classmethod
    def from_dict(cls, variables: dict): 
        _vars = []
        for k, v in variables.items(): 
            _vars.append(Variable(k, v))
        return cls(_vars)    
        
    def __init__(self, variables: List[Variable]): 
        self._variables = variables
        self._N = len(variables)
        self.build_OA()

    def build_OA(self) -> int:
        m_v = None 
        for v in self._variables: 
            if not m_v or v.n_values > m_v.n_values: 
                m_v = v 
        self._Q = m_v.n_values
        self._find_J()
        self._M = self._Q ** self._J
        self._OA = np.full((self._M, self._N), 2, dtype=int)
        self._OA_values = np.full((self._M, self._N), None)
        self._fill_OA()
        return self

    def _fill_OA(self):
        ## For Basic Columns
        for k in range(1, self._J+1): 
            j = int ( ((self._Q**(k-1)) - 1 ) \
                     / (self._Q - 1) ) + 1
            max_i = int(self._Q ** self._J)
            for i in range (1, max_i + 1): 
                den = (self._Q ** (self._J - k))
                # Fixing i,j for pyhton zero based array
                self._OA[i-1, j-1] = int((i - 1) / den) % self._Q #if den > 0 else None

        # For Non-Basic Columns
        for k in range (2, self._J+1): 
            j = int ( ((self._Q ** (k-1)) - 1 ) \
                     / (self._Q - 1) ) + 1
            for s in range (1, j): 
                for t in range (1, self._Q):
                    a_s = self._OA[:,s-1]
                    a_j = self._OA[:,j-1]
                    a_jj = int((j+(s-1)*(self._Q-1)+t)) - 1 
                    if a_jj < self._N:
                        self._OA[:,a_jj] = np.mod(a_s *t  + a_j, self._Q)

        self._OA = self._OA[:,0:self._N]

        for i in range (0, self._M): 
            for j in range(0, self._N): 
                v = self._variables[j]
                vi  = self._OA[i,j]
                try: 
                    value = v.values[vi]
                    self._OA_values[i, j] = value
                except: 
                    print(f"Error at {v}, {vi}, ({i,j})")
    
    def _calc_N_for_QJ(self): 
        return int( (self._Q**(self._J) - 1)/(self._Q-1) )
        
    def _find_J(self): 
        self._J = int(np.log(self._N * (self._Q-1) + 1) / np.log(self._Q))
        n = self._calc_N_for_QJ()
        if n > self._N: 
            self._J -= 1
        elif n < self._N: 
            self._J += 1
        self.N = self._calc_N_for_QJ() 
        
    @property
    def OA(self): 
        return self._OA_values

    def get_params(self, n=0): 
        return { self._variables[i].name : self._OA_values[n][i] for i in range(self._N)}
                
    def __str__(self): 
        return f"(Q = {self._Q}, N  = {self._N}, J = {self._J}, M = {self._M}, N = {self._N})"



# %%
from functools import reduce
def dict_to_level_list(params, substring):
    level_list = {int(k.replace(substring, "", )): v for k, v in params.items() if substring in k}
    level_list = reduce(lambda x, y: x + [y[0]]*y[1], level_list.items(), [])
    return tuple(level_list)
def generate_taguchi(levels_global, levels_local):
    global_level_list = []
    local_level_list = []
    param_dict = {f"global_level_{x}": [0, 1] for x in range(1, levels_global+1)}
    param_dict.update({f"local_level_{x}": [0, 1] for x in range(1, levels_local+1)})
    TgOpt = TaguchiOpt.from_dict(param_dict)
    for e in range(len(TgOpt.OA)): 
        params = TgOpt.get_params(e)
        global_level_list = dict_to_level_list(params, "global_level_")
        local_level_list = dict_to_level_list(params, "local_level_")
        yield global_level_list, local_level_list


# %%

# %%
if __name__ == "__main__":
    for global_level_list, local_level_list  in generate_taguchi(levels_global=6, levels_local=3):
        print(global_level_list, local_level_list)

