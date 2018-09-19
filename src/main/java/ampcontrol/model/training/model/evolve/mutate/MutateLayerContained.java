package ampcontrol.model.training.model.evolve.mutate;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.transferlearning.TransferLearning;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Inserts a layer in the graph. If a layer with the same name exists already it will be replaced. Note that for this
 * to work, the change of layer must be contained, i.e it must not "spill over" to other layers. Examples of what
 * "spills over" is changes in nIn or nOut as well as general changes in output shape in case layer is one of several
 * inputs to a vertex which require all their inputs to be of same shape.
 *
 * @author Christian Skärby
 */
public class MutateLayerContained implements Mutation<TransferLearning.GraphBuilder> {

    private final Supplier<LayerMutation> mutationSupplier;

    public MutateLayerContained(Supplier<LayerMutation> mutationSupplier) {
        this.mutationSupplier = mutationSupplier;
    }

    @Builder
    @Getter
    public static class LayerMutation {
        private final String layerName;
        private final java.util.function.Supplier<Layer.Builder> layerSupplier;
        private final String[] inputLayers;
        private final Function<ComputationGraph, Integer> inputSizeMapping = graph ->
                Optional.ofNullable(graph.getVertex(getLayerName()))
                        .filter(GraphVertex::hasLayer)
                        .map(graphVertex -> graph.layerInputSize(graphVertex.getVertexName()))
                        .orElseGet(
                                () -> Stream.of(getInputLayers())
                                        .mapToInt(graph::layerSize)
                                        .sum());
        private final Function<ComputationGraph, Integer> outputSizeMapping = graph ->
                Optional.ofNullable(graph.getVertex(getLayerName()))
                        .filter(GraphVertex::hasLayer)
                        .map(graphVertex -> graph.layerSize(graphVertex.getVertexName()))
                        .orElseGet(
                                () -> graph.getConfiguration().getVertexInputs().entrySet().stream()
                                        .filter(entry -> entry.getValue().contains(getLayerName()))
                                        .map(Map.Entry::getKey)
                                        .mapToInt(graph::layerInputSize)
                                        .sum());
    }


    @Override
    public TransferLearning.GraphBuilder mutate(TransferLearning.GraphBuilder builder) {
        mutationSupplier.stream().forEach(mutation -> replaceOrAddVertex(mutation, builder, builder.build()));
        return builder;
    }

    private void replaceOrAddVertex(LayerMutation mutation, TransferLearning.GraphBuilder builder, ComputationGraph prevGraph) {

        int nOut = 0;
        if (Optional.ofNullable(prevGraph.getVertex(mutation.getLayerName())).isPresent()) {
            // Simple case, we can replace an existing vertex
            builder.removeVertexKeepConnections(mutation.getLayerName());
            nOut = mutation.getOutputSizeMapping().apply(prevGraph);
        } else {
            // Trickier case: Insert a vertex between two other vertices
            // vertexN -> vertexN+1 need to become vertexN -> newlayer -> vertexN+1
            // Thus, layerN+1 must be removed and added back with newLayer as input
            for (String inputName : mutation.getInputLayers()) {
                for (VertexIndices vertexIndices : prevGraph.getVertex(inputName).getOutputVertices()) {
                    // this is a vertex which shall have the new layer as input
                    final GraphVertex outputVertex = prevGraph.getVertices()[vertexIndices.getVertexIndex()];
                    final String outputName = outputVertex.getVertexName();
                    final org.deeplearning4j.nn.conf.graph.GraphVertex vertexConf = prevGraph.getConfiguration().getVertices().get(outputName);

                    // Handle case of multiple inputs, like this:
                    // {vertex0,..., vertexN} -> outputVertex shall become vertexi -> newlayer and then {vertex0,...,newlayer,...,vertexN}-> outputVertex
                    // Must add back all other vertices as input to outputVertex.
                    final List<String> allInputsToOutput = new ArrayList<>(prevGraph.getConfiguration().getVertexInputs().get(outputName));
                    // Better insert new layer in the same place as the one which is removed...
                    final int index = allInputsToOutput.indexOf(inputName);
                    allInputsToOutput.remove(index);
                    allInputsToOutput.add(index, mutation.layerName);

                    builder.removeVertexKeepConnections(outputVertex.getVertexName())
                            .addVertex(outputName,
                                    vertexConf,
                                    allInputsToOutput.toArray(new String[]{}));

                    // Assumes merge of inputs. Not much else to do without access to TransferLearning builder internals
                    // Note: Will fail in case of a non-layer vertex. Needs fixing...
                    nOut += prevGraph.layerSize(inputName);
                }
            }
        }

        final Layer.Builder layerBuilder = mutation.getLayerSupplier().get();
        if (layerBuilder instanceof FeedForwardLayer.Builder) {
            FeedForwardLayer.Builder ffBuilder = ((FeedForwardLayer.Builder) layerBuilder);
            ffBuilder.nIn(mutation.getInputSizeMapping().apply(prevGraph));
            ffBuilder.nOut(nOut);
        }

        builder.addLayer(mutation.getLayerName(), layerBuilder.build(), mutation.getInputLayers());
    }


}
