package ampcontrol.model.training.model.mutate;

import ampcontrol.model.training.model.mutate.reshape.ReshapeRegistry;
import ampcontrol.model.training.model.mutate.reshape.ReshapeTask;
import ampcontrol.model.training.model.mutate.reshape.SingleReshapeSubTask;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Comparator;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

public class MutationGraph {

    private final ComputationGraph graph;
    private final Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory;

    public MutationGraph(ComputationGraph graph) {
        this(graph, str -> Optional.empty());
    }

    public MutationGraph(ComputationGraph graph, Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory) {
        this.graph = graph;
        this.compFactory = compFactory;
    }

    /**
     * Mutate the current graph into a new {@link ComputationGraphConfiguration} by transferring parameters
     *
     * @param config configuration of the new graph
     * @return A {@link ComputationGraph} build from the given config initialized with parameters from the existing graph
     */
    public ComputationGraph mutateTo(ComputationGraphConfiguration config) {

        final ComputationGraph newGraph = new ComputationGraph(config);
        newGraph.init();

        final ReshapeRegistry registry = new ReshapeRegistry();
        final int[] topologicalOrder = newGraph.topologicalSortOrder();
        final GraphVertex[] vertices = newGraph.getVertices();
        //set params from orig graph as necessary to new graph
        for (int vertexIndex : topologicalOrder) {
            transferParameters(registry, newGraph, vertices[vertexIndex].getVertexName());
        }
        registry.commit();
        return newGraph;
    }

    private Optional<GraphVertex> findLayerVertex(String name, ComputationGraph graph) {
        return Stream.of(graph.getVertices())
                .filter(GraphVertex::hasLayer)
                .filter(vertex -> vertex.getLayer().numParams() > 0)
                .filter(vertex -> name.equals(vertex.getVertexName()))
                .findAny();
    }

    private void transferParameters(ReshapeRegistry registry, ComputationGraph newGraph, String layerName) {
        Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
        Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

            final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
            final Map<String, INDArray> targetParams = targetVertex.paramTable(false);

            ReshapeTask.Builder taskBuilder = ReshapeTask.builder()
                    .sourceShape(sourceParams.get(DefaultParamInitializer.WEIGHT_KEY).shape())
                    .targetShape(targetParams.get(DefaultParamInitializer.WEIGHT_KEY).shape());

            SingleReshapeSubTask.Builder reshapeListBuilder = initReshapeListBuilder(registry, layerName, sourceParams, targetParams);

            Optional.ofNullable(targetVertex.getOutputVertices())
                    .ifPresent(vertexIndices -> {
                        for (VertexIndices vertexIndex : vertexIndices) {
                            final String name = newGraph.getVertices()[vertexIndex.getVertexIndex()].getVertexName();
                            transferOutputParameters(registry, newGraph, name, reshapeListBuilder);
                        }
                    });

            taskBuilder.reshapeSubTask(reshapeListBuilder.build())
                    .build()
                    .reshape();
        }
    }

    private SingleReshapeSubTask.Builder initReshapeListBuilder(
            ReshapeRegistry registry,
            String layerName,
            Map<String, INDArray> sourceParams,
            Map<String, INDArray> targetParams) {

        final SingleReshapeSubTask.Builder firstSubtaskBuilder = SingleReshapeSubTask.builder()
                .maskDim(1) // Dim 1 is input dim. Shall be transferred based on cropped outputs from previous layer
                .source(SingleReshapeSubTask.IndMapping.builder()
                        .entry(registry.register(sourceParams.get(DefaultParamInitializer.WEIGHT_KEY), layerName + "_source_W"))
                        .build())
                .target(SingleReshapeSubTask.IndMapping.builder()
                        .entry(registry.register(targetParams.get(DefaultParamInitializer.WEIGHT_KEY), layerName + "_target_W"))
                        .build());

        compFactory.apply(layerName)
                .ifPresent(firstSubtaskBuilder::compFactory);

        return firstSubtaskBuilder
                .addDependentTask(SingleReshapeSubTask.builder()
                        .source(SingleReshapeSubTask.IndMapping.builder()
                                .entry(registry.register(sourceParams.get(DefaultParamInitializer.BIAS_KEY), layerName + "_source_b"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : 1)
                                .build())
                        .target(SingleReshapeSubTask.IndMapping.builder()
                                .entry(registry.register(targetParams.get(DefaultParamInitializer.BIAS_KEY), layerName + "_target_b"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim)
                                .build())
                        );
    }

    private void transferOutputParameters(
            ReshapeRegistry registry,
            ComputationGraph newGraph,
            String layerName,
            SingleReshapeSubTask.Builder sublistBuilder) {
        Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
        Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

            final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
            final Map<String, INDArray> targetParams = targetVertex.paramTable(false);
            final INDArray sourceParam = sourceParams.get(DefaultParamInitializer.WEIGHT_KEY);
            final INDArray targetParam = targetParams.get(DefaultParamInitializer.WEIGHT_KEY);

            sublistBuilder.addDependentTask(SingleReshapeSubTask.builder()
                    .source(SingleReshapeSubTask.IndMapping.builder()
                            .entry(registry.register(sourceParam, layerName + "_source_W"))
                            .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim) // flip dim 0 and 1
                            .build())
                    .target(SingleReshapeSubTask.IndMapping.builder()
                            .entry(registry.register(targetParam, layerName + "_target_W"))
                            .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim) // flip dim 0 and 1
                            .build()));
        }

    }
}
