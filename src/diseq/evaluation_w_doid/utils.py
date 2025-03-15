import pronto
import networkx as nx

# Load disease ontology from OBO file from https://github.com/DiseaseOntology/HumanDiseaseOntology
def load_disease_ontology(obo_file="./data/doid.obo"):
    return pronto.Ontology(obo_file)

# Build a directed ontology graph from parent-child relationships
def build_ontology_graph(ontology):
    G = nx.DiGraph()
    
    for term in ontology.terms.values():
        if term.id.startswith("DOID:"):
            G.add_node(term.id)  # Ensure all terms exist as nodes
            for parent in term.parents:
                if parent.id.startswith("DOID:"):
                    G.add_edge(parent.id, term.id)
    
    return G

# Convert ontology graph to an undirected graph
def convert_to_undirected(graph):
    return graph.to_undirected()

# Compute distance using shortest path using ontology relationships
def compute_doid_distance(graph, doid1, doid2):
    if not doid1 or not doid2:
        return None

    doid1, doid2 = doid1.strip(), doid2.strip()

    if doid1 not in graph:
        print(f"DOID {doid1} not found in ontology")
        return None

    if doid2 not in graph:
        print(f"DOID {doid2} not found in ontology")
        return None

    try:
        return nx.shortest_path_length(graph, source=doid1, target=doid2)
    except nx.NetworkXNoPath:
        print(f"No path found between {doid1} and {doid2}")
        return None
    

# Check missing distance reasons (usually due to "control" cases as an original label)
def get_missing_reason(graph, doid1, doid2):
    if not doid1 or not doid2:
        return "Missing DOID"
    
    if doid1 not in graph:
        return f"{doid1} not in Ontology"
    
    if doid2 not in graph:
        return f"{doid2} not in Ontology"
    
    if not nx.has_path(graph, doid1, doid2):
        return "No Path (Disconnected Component)"
    
    return "Unknown Issue"