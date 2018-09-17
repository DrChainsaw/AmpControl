package ampcontrol.model.training.model.evolve.mutate;

import lombok.Builder;
import lombok.Getter;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.transferlearning.TransferLearning;

import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Inserts a layer in the graph. If a layer with the same name exists already it will be replaced. Note that for this
 * to work, the change of layer must be contained, i.e it must not "spill over" to other layers. Examples of what
 * "spills over" is changes in nIn or nOut as well as general changes in output shape in case layer is one of several
 * inputs to a vertex which require all their inputs to be of same shape.
 *
 * @author Christian Skärby
 */
public class MutateLayerContained implements Mutation {

    private final Supplier<Stream<LayerMutation>> mutationSupplier;

    public MutateLayerContained(Supplier<Stream<LayerMutation>> mutationSupplier) {
        this.mutationSupplier = mutationSupplier;
    }

    @Builder
    @Getter
    public static class LayerMutation {
        private final String layerName;
        private final Supplier<Layer.Builder> layerSupplier;
        private final String[] inputLayers;
        private final Function<ComputationGraph, Integer> inputSizeMapping = graph ->
                Optional.ofNullable(graph.getVertex(getLayerName()))
                        .filter(GraphVertex::hasLayer)
                        .map(graphVertex -> graph.layerInputSize(graphVertex.getVertexName()))
                        .orElseGet(
                                () -> Stream.of(getInputLayers())
                                        .mapToInt(graph::layerSize)
                                        .sum());
    }


    @Override
    public TransferLearning.GraphBuilder mutate(TransferLearning.GraphBuilder builder, ComputationGraph prevGraph) {
        mutationSupplier.get().forEach(mutation -> replaceOrAddVertex(mutation, builder, prevGraph));
        return builder;
    }

    private void replaceOrAddVertex(LayerMutation mutation, TransferLearning.GraphBuilder builder, ComputationGraph prevGraph) {

        if(Optional.ofNullable(prevGraph.getVertex(mutation.getLayerName())).isPresent())  {
            builder.removeVertexKeepConnections(mutation.getLayerName());
        } else {
            // layerN -> layerN+1 need to become layerN -> newlayer -> layerN+1
            // Thus, layerN+1 must be removed and added back with newLayer as input
            for(String inputName: mutation.getInputLayers()) {
                for(VertexIndices vertexIndices: prevGraph.getVertex(inputName).getOutputVertices()) {
                    // this is a vertex which shall have the new layer as input
                    final GraphVertex outputVertex = prevGraph.getVertices()[vertexIndices.getVertexIndex()];
                    final String outputName = outputVertex.getVertexName();
                    final org.deeplearning4j.nn.conf.graph.GraphVertex vertexConf = prevGraph.getConfiguration().getVertices().get(outputName);

                    // Handle case of multiple inputs, like this:
                    // {layerA, layerB} -> layerC shall become layerA -> newlayer and then {newlayer,layerB}-> layerC
                    // Must add back layerB as input to layerC (outputVertex is layerC here).
                    final List<String> allInputsToOutput = prevGraph.getConfiguration().getVertexInputs().get(outputName).stream()
                            .filter(layername -> !inputName.equals(layername)) //inputName is the one we want to replace with mutation.layerName;
                            .collect(Collectors.toList());
                    allInputsToOutput.add(mutation.layerName);

                    builder.removeVertexKeepConnections(outputVertex.getVertexName())
                            .addVertex(outputName,
                                    vertexConf,
                                    allInputsToOutput.toArray(new String[] {}));
                }
            }
        }

        final Layer.Builder layerBuilder = mutation.getLayerSupplier().get();
        if (layerBuilder instanceof FeedForwardLayer.Builder) {
            ((FeedForwardLayer.Builder) layerBuilder).nIn(mutation.inputSizeMapping.apply(prevGraph));
        }

        builder.addLayer(mutation.getLayerName(), layerBuilder.build(), mutation.getInputLayers());
    }


}
