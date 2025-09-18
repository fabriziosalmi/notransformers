import torch
import torch.nn as nn
import random
from primitives import InputNode, Linear, ElementwiseAdd, Concatenate, GatedRecurrentUnit, ExponentialMovingAverage, Conv1D, LayerNorm, GatedLinearUnit, Activation

class ModelGenome(nn.Module):
    def __init__(self, nodes, connections, learning_params=None):
        super().__init__()
        self.nodes = nn.ModuleList(nodes)
        self.connections = connections  # list of (src_idx, dst_idx, dst_input_slot)
        if learning_params is None:
            learning_params = {'learning_rate': 0.01, 'optimizer': 'adam', 'activation_function': 'relu'}
        self.learning_params = learning_params

    def is_valid(self):
        # 1. Controllo di connessione: tutti i nodi tranne InputNode devono avere almeno un input
        input_nodes = [i for i, n in enumerate(self.nodes) if isinstance(n, InputNode)]
        input_set = set(input_nodes)
        incoming = {i: 0 for i in range(len(self.nodes))}
        for src, dst, slot in self.connections:
            if 0 <= dst < len(self.nodes):
                incoming[dst] += 1
        for i, n in enumerate(self.nodes):
            if i not in input_set and incoming[i] == 0:
                return False
        
        # 2. Controllo percorso: esiste un cammino da InputNode a output (ultimo nodo)
        output_idx = len(self.nodes) - 1
        visited = set()
        queue = [output_idx]
        while queue:
            curr = queue.pop(0)
            visited.add(curr)
            for src, dst, slot in self.connections:
                if dst == curr and 0 <= src < len(self.nodes) and src not in visited:
                    queue.append(src)
        if not any(i in visited for i in input_nodes):
            return False
        
        # 3. Controllo dimensioni Linear: input/output compatibili
        node_outputs = {}
        for idx, node in enumerate(self.nodes):
            if isinstance(node, InputNode):
                node_outputs[idx] = getattr(node, 'input_dim', None)
        
        for src, dst, slot in self.connections:
            if 0 <= src < len(self.nodes) and 0 <= dst < len(self.nodes):
                src_node = self.nodes[src]
                dst_node = self.nodes[dst]
                
                if isinstance(src_node, Linear):
                    out_dim = src_node.output_dim
                elif isinstance(src_node, InputNode):
                    out_dim = src_node.input_dim
                else:
                    out_dim = node_outputs.get(src, None)
                
                if isinstance(dst_node, Linear):
                    if out_dim is not None and out_dim != dst_node.input_dim:
                        return False
                
                node_outputs[dst] = out_dim
        return True

    def forward(self, sequence):
        # sequence: shape (batch, seq_len, input_dim) torch tensor
        batch, seq_len, _ = sequence.shape
        
        # Reset stato di tutti i nodi
        for node in self.nodes:
            if hasattr(node, 'reset'):
                node.reset()
        
        node_outputs = {}
        node_states = {}
        indegree = {i: 0 for i in range(len(self.nodes))}
        input_map = {i: [] for i in range(len(self.nodes))}
        
        for src, dst, slot in self.connections:
            indegree[dst] += 1
            input_map[dst].append((slot, src))
        
        # Topological sort
        ready = [i for i, n in enumerate(self.nodes) if indegree[i] == 0]
        order = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for src, dst, slot in self.connections:
                if src == n:
                    indegree[dst] -= 1
                    if indegree[dst] == 0:
                        ready.append(dst)
        
        # Per ogni passo temporale
        for t in range(seq_len):
            for idx in order:
                node = self.nodes[idx]
                
                # InputNode prende il valore della sequenza al tempo t
                if isinstance(node, InputNode):
                    out = node.forward(sequence[:, t, :])
                    node_outputs[idx] = out
                else:
                    # Raccogli input
                    if input_map[idx]:
                        inputs = [node_outputs[src] for _, src in sorted(input_map[idx], key=lambda t: t[0])]
                    else:
                        inputs = []
                    
                    try:
                        # Prova prima con tutti gli input
                        if len(inputs) == 0:
                            # Nodo senza input (impossibile ma per sicurezza)
                            result = torch.zeros(batch, 1, device=sequence.device)
                        else:
                            result = node.forward(*inputs)
                    except TypeError as e:
                        # Per nodi che richiedono un numero specifico di input
                        if isinstance(node, (ElementwiseAdd, Concatenate)):
                            if len(inputs) == 1:
                                # Duplica l'input per nodi che richiedono 2 input
                                result = node.forward(inputs[0], inputs[0])
                            elif len(inputs) >= 2:
                                result = node.forward(inputs[0], inputs[1])
                            else:
                                result = torch.zeros(batch, 1, device=sequence.device)
                        elif len(inputs) == 1:
                            result = node.forward(inputs[0])
                        else:
                            result = torch.zeros(batch, 1, device=sequence.device)
                    except Exception as e:
                        # Debug per capire meglio gli errori
                        print(f"Error in node {idx} ({type(node)}): {e}")
                        print(f"Input shapes: {[inp.shape if hasattr(inp, 'shape') else type(inp) for inp in inputs]}")
                        raise e
                    
                    node_outputs[idx] = result
        
        return node_outputs[order[-1]]

    @staticmethod
    def create_random_genome(input_dim, output_dim, max_nodes=10):
        nodes = []
        connections = []
        learning_rate = 10 ** random.uniform(-4, -2)
        optimizer = random.choice(['sgd', 'adam'])
        activation_function = random.choice(['relu', 'tanh', 'sigmoid'])
        lr_scheduler = random.choice(['none', 'step', 'cosine', 'exponential'])
        learning_params = {
            'learning_rate': learning_rate,
            'optimizer': optimizer,
            'activation_function': activation_function,
            'lr_scheduler': lr_scheduler
        }
        
        nodes.append(InputNode(input_dim))
        n_hidden = random.randint(1, max_nodes-2)
        primitive_choices = [Linear, ElementwiseAdd, GatedRecurrentUnit, ExponentialMovingAverage, Conv1D, LayerNorm, GatedLinearUnit, Concatenate, Activation]
        
        for i in range(n_hidden):
            prim_type = random.choice(primitive_choices)
            if prim_type == Linear:
                in_dim = input_dim
                out_dim = random.choice([input_dim, output_dim])
                nodes.append(Linear(in_dim, out_dim))
            elif prim_type == ElementwiseAdd:
                nodes.append(ElementwiseAdd())
            elif prim_type == GatedRecurrentUnit:
                hid_dim = random.choice([input_dim, output_dim])
                nodes.append(GatedRecurrentUnit(input_dim, hid_dim))
            elif prim_type == ExponentialMovingAverage:
                nodes.append(ExponentialMovingAverage(input_dim))
            elif prim_type == Conv1D:
                nodes.append(Conv1D(input_dim, output_dim, kernel_size=3))
            elif prim_type == LayerNorm:
                nodes.append(LayerNorm(input_dim))
            elif prim_type == GatedLinearUnit:
                nodes.append(GatedLinearUnit(input_dim, output_dim))
            elif prim_type == Concatenate:
                nodes.append(Concatenate())
            elif prim_type == Activation:
                nodes.append(Activation(kind=activation_function))
        
        nodes.append(Linear(input_dim, output_dim))
        
        for i in range(1, len(nodes)-1):
            connections.append((i-1, i, 0))
        connections.append((len(nodes)-2, len(nodes)-1, 0))
        
        return ModelGenome(nodes, connections, learning_params)

    def __repr__(self):
        s = "ModelGenome (PyTorch):\n"
        for i, n in enumerate(self.nodes):
            s += f"  [{i}] {n}\n"
        s += "Connections:\n"
        for c in self.connections:
            s += f"  {c}\n"
        return s

    def mutate_add_node(self):
        """Aggiunge un nuovo nodo random nel genoma"""
        primitive_choices = [Linear, ElementwiseAdd, GatedRecurrentUnit, ExponentialMovingAverage, 
                           Conv1D, LayerNorm, GatedLinearUnit, Concatenate, Activation]
        prim_type = random.choice(primitive_choices)
        
        # Crea il nuovo nodo con parametri casuali
        input_dim = getattr(self.nodes[0], 'input_dim', 1)  # Prende da InputNode
        output_dim = random.choice([1, 2, 4, 8, 16])
        
        if prim_type == Linear:
            new_node = Linear(input_dim, output_dim)
        elif prim_type == ElementwiseAdd:
            new_node = ElementwiseAdd()
        elif prim_type == GatedRecurrentUnit:
            new_node = GatedRecurrentUnit(input_dim, output_dim)
        elif prim_type == ExponentialMovingAverage:
            new_node = ExponentialMovingAverage(input_dim)
        elif prim_type == Conv1D:
            new_node = Conv1D(input_dim, output_dim, kernel_size=3)
        elif prim_type == LayerNorm:
            new_node = LayerNorm(input_dim)
        elif prim_type == GatedLinearUnit:
            new_node = GatedLinearUnit(input_dim, output_dim)
        elif prim_type == Concatenate:
            new_node = Concatenate()
        elif prim_type == Activation:
            activation_kind = random.choice(['relu', 'tanh', 'sigmoid'])
            new_node = Activation(kind=activation_kind)
        
        # Inserisce il nodo prima dell'ultimo (che è l'output)
        insert_idx = len(self.nodes) - 1
        self.nodes.insert(insert_idx, new_node)
        
        # Aggiorna gli indici delle connessioni esistenti
        updated_connections = []
        for src, dst, slot in self.connections:
            new_src = src if src < insert_idx else src + 1
            new_dst = dst if dst < insert_idx else dst + 1
            updated_connections.append((new_src, new_dst, slot))
        
        # Aggiunge una connessione casuale al nuovo nodo
        if insert_idx > 0:
            src_node = random.randint(0, insert_idx - 1)
            updated_connections.append((src_node, insert_idx, 0))
        
        self.connections = updated_connections

    def mutate_add_connection(self, residual_probability=0.5):
        """Aggiunge una nuova connessione, possibilmente residua"""
        if len(self.nodes) < 3:
            return  # Troppo pochi nodi
        
        # Decide se fare una connessione residua
        if random.random() < residual_probability:
            self._add_residual_connection()
        else:
            self._add_normal_connection()

    def _add_normal_connection(self):
        """Aggiunge una connessione normale"""
        max_attempts = 10
        for _ in range(max_attempts):
            src = random.randint(0, len(self.nodes) - 2)
            dst = random.randint(src + 1, len(self.nodes) - 1)
            slot = 0
            
            # Controlla se la connessione esiste già
            if (src, dst, slot) not in self.connections:
                self.connections.append((src, dst, slot))
                break

    def _add_residual_connection(self):
        """Implementa una connessione residua A -> Add <- B"""
        if len(self.nodes) < 4:  # Serve almeno Input, A, B, Output
            return
        
        max_attempts = 20
        for _ in range(max_attempts):
            # Scegli nodo A (deve essere prima di B)
            node_a_idx = random.randint(0, len(self.nodes) - 3)
            # Scegli nodo B (deve essere dopo A, ma non l'ultimo che è output)
            node_b_idx = random.randint(node_a_idx + 1, len(self.nodes) - 2)
            
            # Estima le dimensioni di output di A e B
            dim_a = self._estimate_output_dim(node_a_idx)
            dim_b = self._estimate_output_dim(node_b_idx)
            
            if dim_a is None or dim_b is None:
                continue
            
            # Se le dimensioni sono diverse, inserisci un Linear adapter
            adapter_node = None
            adapter_idx = None
            
            if dim_a != dim_b:
                # Crea un nodo Linear per adattare A alle dimensioni di B
                adapter_node = Linear(dim_a, dim_b)
                adapter_idx = len(self.nodes)
                self.nodes.append(adapter_node)
            
            # Crea il nodo ElementwiseAdd
            add_node = ElementwiseAdd()
            add_idx = len(self.nodes)
            self.nodes.append(add_node)
            
            # Aggiorna le connessioni esistenti che puntavano a B
            updated_connections = []
            for src, dst, slot in self.connections:
                if dst == node_b_idx:
                    # Le connessioni a B ora vanno al nodo Add
                    if src == node_a_idx:
                        # Evita duplicati se A era già connesso a B
                        continue
                    updated_connections.append((src, add_idx, 0))  # Primo input dell'Add
                else:
                    updated_connections.append((src, dst, slot))
            
            # Connessioni per la struttura residua
            if adapter_idx is not None:
                # A -> Adapter -> Add (secondo input)
                updated_connections.append((node_a_idx, adapter_idx, 0))
                updated_connections.append((adapter_idx, add_idx, 1))
            else:
                # A -> Add (secondo input) direttamente
                updated_connections.append((node_a_idx, add_idx, 1))
            
            # B -> Add (primo input)
            updated_connections.append((node_b_idx, add_idx, 0))
            
            self.connections = updated_connections
            
            print(f"Residual connection added: A={node_a_idx} -> Add={add_idx} <- B={node_b_idx}")
            if adapter_idx is not None:
                print(f"  With adapter: A={node_a_idx} -> Adapter={adapter_idx} -> Add={add_idx}")
            break

    def _estimate_output_dim(self, node_idx):
        """Stima la dimensione di output di un nodo"""
        node = self.nodes[node_idx]
        
        if isinstance(node, InputNode):
            return node.input_dim
        elif isinstance(node, Linear):
            return node.output_dim
        elif isinstance(node, GatedRecurrentUnit):
            return node.output_dim
        elif isinstance(node, Conv1D):
            return node.output_dim
        elif isinstance(node, GatedLinearUnit):
            return node.output_dim
        elif isinstance(node, (ElementwiseAdd, Activation, LayerNorm, ExponentialMovingAverage)):
            # Questi mantengono la dimensione dell'input
            # Trova il predecessore e usa la sua dimensione
            for src, dst, slot in self.connections:
                if dst == node_idx:
                    return self._estimate_output_dim(src)
            return None
        elif isinstance(node, Concatenate):
            # Somma le dimensioni degli input
            total_dim = 0
            for src, dst, slot in self.connections:
                if dst == node_idx:
                    dim = self._estimate_output_dim(src)
                    if dim is not None:
                        total_dim += dim
            return total_dim if total_dim > 0 else None
        else:
            return None

    def mutate_remove_connection(self):
        """Rimuove una connessione casuale (mantenendo validità)"""
        if len(self.connections) <= len(self.nodes) - 1:
            return  # Troppo poche connessioni
        
        removable = []
        for i, (src, dst, slot) in enumerate(self.connections):
            # Simula la rimozione e controlla se il grafo rimane valido
            temp_connections = self.connections[:i] + self.connections[i+1:]
            temp_genome = ModelGenome(self.nodes, temp_connections, self.learning_params)
            if temp_genome.is_valid():
                removable.append(i)
        
        if removable:
            remove_idx = random.choice(removable)
            self.connections.pop(remove_idx)

    def mutate_learning_params(self):
        """Muta i parametri di apprendimento"""
        param = random.choice(['learning_rate', 'optimizer', 'lr_scheduler'])
        if param == 'learning_rate':
            self.learning_params['learning_rate'] = 10 ** random.uniform(-4, -2)
        elif param == 'optimizer':
            self.learning_params['optimizer'] = random.choice(['sgd', 'adam'])
        elif param == 'lr_scheduler':
            self.learning_params['lr_scheduler'] = random.choice(['none', 'step', 'cosine', 'exponential'])

    def mutate(self, mutation_rate=0.1):
        """Applica mutazioni casuali al genoma"""
        mutations = [
            self.mutate_add_node,
            self.mutate_add_connection,
            self.mutate_remove_connection,
            self.mutate_learning_params
        ]
        
        for mutation in mutations:
            if random.random() < mutation_rate:
                mutation()