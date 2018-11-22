package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;

import java.util.Optional;
import java.util.stream.Stream;

/**
 * {@link Graph} in backwards direction (output -> input) for a {@link ComputationGraph}.
 *
 * @author Christian Skärby
 */
public class BackwardOfCompGraph implements Graph<String> {

    private final ComputationGraph graph;

    BackwardOfCompGraph(ComputationGraph graph) {
        this.graph = graph;
    }

    @Override
    public Stream<String> children(String vertex) {
        return  Stream.of(Optional.ofNullable(graph.getVertex(vertex).getInputVertices()).orElse(new VertexIndices[0]))
                .map(vertexIndex -> graph.getVertices()[vertexIndex.getVertexIndex()])
                .map(GraphVertex::getVertexName);
    }
}
