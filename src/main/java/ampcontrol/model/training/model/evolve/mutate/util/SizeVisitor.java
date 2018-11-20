package ampcontrol.model.training.model.evolve.mutate.util;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Calculates size of visited vertices. Main purpose is to split up sizes when passing through a {@link MergeVertex}
 *
 * @author Christian Skärby
 */
public class SizeVisitor {

    private final Graph<String> graph;
    private final BiFunction<Long, Long, Long> truncate;
    private final ComputationGraphConfiguration.GraphBuilder builder;
    private final long startSize;
    private final Map<String, Long> visitedToSize = new HashMap<>();

    public SizeVisitor(
            Graph<String> graph,
            ComputationGraphConfiguration.GraphBuilder builder,
            long startSize,
            BiFunction<Long, Long, Long> truncate) {
        this.graph = graph;
        this.builder = builder;
        this.startSize = startSize;
        this.truncate = truncate;
    }

    /**
     * Set the given vertex to the given value. Typically used for initialization
     * @param vertex vertex to set
     * @param size value to set
     */
    public void set(String vertex, long size) {
        visitedToSize.put(vertex, size);
    }

    /**
     * Determine the size of any ancestor vertices
     * @param vertex vertex to visit
     */
    public void visit(String vertex) {

        final List<String> inputs = graph.children(vertex).collect(Collectors.toList());

        if(inputs.stream().allMatch(visitedToSize::containsKey)) {
            return;
        }

        // TODO: This will break easily. Better to provide some "shallSplit" predicate which depends on how the graph is configured
        if (inputs.size() > 1
                && !(builder.getVertices().get(vertex) instanceof ElementWiseVertex)) {
            long remainder = visitedToSize.getOrDefault(vertex, startSize);
            final long[] layerSizes = new long[inputs.size()];
            final Boolean[] validLayers = new Boolean[inputs.size()];

            for (int i = 0; i < validLayers.length; i++) {
                final String inputName = inputs.get(i);
                layerSizes[i] = GraphBuilderUtil.getOutputSize(inputName, builder);
                validLayers[i] = layerSizes[i] > 0;
                if (validLayers[i]) {
                    remainder -= visitedToSize.getOrDefault(inputName, 0L);
                }
            }

            final BinaryOperator<Long> maxOrMin = startSize > 0 ? Math::min : Math::max;
            for (int i = 0; i < validLayers.length; i++) {
                final String inputName = inputs.get(i);
                final long layerSizesSum = Arrays.stream(layerSizes, i, validLayers.length).sum();
                if (validLayers[i] && !visitedToSize.containsKey(inputName)) {
                    final long delta = truncate.apply(layerSizes[i], maxOrMin.apply((remainder * layerSizes[i]) / layerSizesSum, remainder));
                    visitedToSize.put(inputs.get(i), delta);
                    remainder -= delta;
                } else if (!validLayers[i]) {
                    visitedToSize.put(inputs.get(i), 0L);
                }
            }

            if (Stream.of(validLayers).anyMatch(valid -> valid) && remainder != 0) {
                throw new IllegalStateException("Failed to distribute size over " + inputs + " sizes: " +
                        visitedToSize + " layerSizes : " + Arrays.toString(layerSizes) + " remainder: " + remainder);
            }

        } else {
            visitedToSize.putAll(inputs.stream()
                    .filter(input -> !visitedToSize.containsKey(input))
                    .collect(Collectors.toMap(
                    name -> name,
                    name -> visitedToSize.getOrDefault(vertex, startSize)
            )));
        }
    }

    /**
     * Return the size of a visited vertex
     * @param vertex Vertex (input vertex must have been visited)
     * @return The size of the vertex
     */
    public Long getSize(String vertex) {
        return visitedToSize.get(vertex);
    }
}
