package ampcontrol.model.training.model.mutate;

import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;

import java.util.Objects;
import java.util.function.IntUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Stream;

/**
 * Mutation operation which changes nOut of layers
 *
 * @author Christian Skärby
 */
public class MutateNout {

    private final Supplier<Stream<String>> mutationLayerSupplier;
    private final IntUnaryOperator mutationFunction;

    public MutateNout(Supplier<Stream<String>> mutationLayerSupplier, IntUnaryOperator mutationFunction) {
        this.mutationLayerSupplier = mutationLayerSupplier;
        this.mutationFunction = mutationFunction;
    }


    public TransferLearning.GraphBuilder mutate(TransferLearning.GraphBuilder builder, ComputationGraph prevGraph) {
        return mutationLayerSupplier.get().reduce(builder, (b, layername) -> mutateBuilder(b, prevGraph, layername), (b1,b2) -> b2);
    }

    private TransferLearning.GraphBuilder mutateBuilder(TransferLearning.GraphBuilder builder, ComputationGraph prevGraph, String layerName) {
        final ComputationGraph tmpGraph = builder.nOutReplace(layerName, mutateNout(prevGraph, layerName), WeightInit.ZERO).build();
        return new TransferLearning.GraphBuilder(tmpGraph);
    }

    private int mutateNout(ComputationGraph prevGraph, String layerName) {
        GraphVertex vertex = Objects.requireNonNull(prevGraph.getVertex(layerName));

        if (!vertex.hasLayer()) {
            throw new IllegalArgumentException("Vertex " + layerName + " has no layer!");
        }

        final Layer layer = vertex.getLayer().conf().getLayer();
        if (layer instanceof FeedForwardLayer) {
            return mutationFunction.applyAsInt((int) ((FeedForwardLayer) layer).getNOut());
        }
        throw new IllegalArgumentException("Can not mutate Nout of layer: " + layer);

    }
}
