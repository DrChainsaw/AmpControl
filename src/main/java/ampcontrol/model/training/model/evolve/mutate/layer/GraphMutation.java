package ampcontrol.model.training.model.evolve.mutate.layer;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.GraphVertex;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * Modifies a {@link GraphBuilder} based on an input {@link GraphMutationDescription}. Useful for adding layers/vertices.
 * Note that in case more than one layer shall be added, the name returned from {@link InputsAndOutputNames#getOutputNames()}
 * needs to be the last vertex added and {@link InputsAndOutputNames#getInputNames()}  needs to be the inputs to the first
 * vertex. {@code NOut} provided to {@link GraphMutationDescription#mutation} is also giving the input size of the
 * first vertex and the output size of the last vertex.
 *
 * @author Christian Skärby
 */
public class GraphMutation implements Mutation<GraphBuilder> {

    private final Supplier<GraphMutationDescription> mutationSupplier;

    @Getter
    @Builder
    public static class GraphMutationDescription {
        private final Function<GraphBuilder, InputsAndOutputNames> mutation;
    }

    @Builder
    @Getter
    public static class InputsAndOutputNames {
        @Singular private final List<String> outputNames;
        @Builder.Default private final Predicate<String> keepInputConnection = str -> true;
        @Singular private final List<String> inputNames;
    }

    public GraphMutation(Supplier<GraphMutationDescription> mutationSupplier) {
        this.mutationSupplier = mutationSupplier;
    }

    @Override
    public ComputationGraphConfiguration.GraphBuilder mutate(ComputationGraphConfiguration.GraphBuilder toMutate) {
        mutationSupplier.stream().forEach(mutation -> {
            final InputsAndOutputNames inputsAndOutputNames = mutation.mutation.apply(toMutate);
            reconnectVertices(inputsAndOutputNames, toMutate);
        });
        return toMutate;
    }

    /**
     * Make room for inserting a vertex between two other vertices. Note that actual addition of "newLayer" in examples
     * below does not happen in this function.
     * vertexN -> vertexN+1 need to become vertexN -> newlayer -> vertexN+1 Thus, layerN+1 must be removed and added
     * back with newLayer as input.
     * <br><br>
     * Also handles case of multiple inputs, like this: <br>
     * {vertex0,..., vertexN} -> outputVertex shall become vertexi -> newlayer followed by
     * {vertex0,...,newlayer,...,vertexN} -> outputVertex. Must remove outputVertex and then add back all other vertices
     * as input to outputVertex along with newLayer.
     *
     * @param mutation Defines how to change the builder
     * @param builder  {@link ComputationGraphConfiguration.GraphBuilder} to mutate
     */
    private static void reconnectVertices(InputsAndOutputNames mutation, ComputationGraphConfiguration.GraphBuilder builder) {
        for (String inputName : mutation.getInputNames()) {

            // Find all layers to which vertex of inputName is output
            final List<String> allOutputLayers = builder.getVertexInputs().entrySet().stream()
                    .filter(layerToInputsEntry -> layerToInputsEntry.getValue().contains(inputName))
                    .map(Map.Entry::getKey)
                    .filter(mutation.getKeepInputConnection().negate())
                    .collect(Collectors.toList());

            for (String outputName : allOutputLayers) {

                final GraphVertex vertexConf = builder.getVertices().get(outputName);

                // Handle case of multiple inputs, like this:
                // {vertex0,..., vertexN} -> outputVertex shall become vertexi -> newlayer and then {vertex0,...,newlayer,...,vertexN}-> outputVertex
                // Must add back all other vertices as input to outputVertex.
                final List<String> allInputsToOutput = new ArrayList<>(builder.getVertexInputs().get(outputName));
                // Better insert new layer in the same place as the one which is removed...
                final int index = allInputsToOutput.indexOf(inputName);
                allInputsToOutput.remove(index);
                int cnt = 0;
                for(String newOutputName: mutation.getOutputNames()) {
                    allInputsToOutput.add(index + cnt, newOutputName);
                    cnt++;
                }
                builder.removeVertex(outputName, false)
                        .addVertex(outputName,
                                vertexConf,
                                allInputsToOutput.toArray(new String[]{}));


            }
        }
    }
}
