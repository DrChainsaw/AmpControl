package ampcontrol.model.training.model.evolve.transfer;

import ampcontrol.model.training.model.evolve.mutate.Mutation;
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

public class WeightTransfer {

    private final ComputationGraph graph;
    private final Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory;

    public WeightTransfer(ComputationGraph graph) {
        this(graph, str -> Optional.empty());
    }

    public WeightTransfer(ComputationGraph graph, Function<String, Optional<Function<int[], Comparator<Integer>>>> compFactory) {
        this.graph = graph;
        this.compFactory = compFactory;
    }

    /**
     * Mutate the current graph into a new {@link ComputationGraphConfiguration} by transferring parameters
     *
     * @param newGraph new computation graph for which weights shall be transferred
     * @return A {@link ComputationGraph} initialized with parameters from the existing graph
     */
    public ComputationGraph mutateTo(ComputationGraph newGraph) {

        final TransferRegistry registry = new TransferRegistry();
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

    private void transferParameters(TransferRegistry registry, ComputationGraph newGraph, String layerName) {
        Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
        Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

        if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
            final GraphVertex sourceVertex = sourceVertexMaybe.get();
            final GraphVertex targetVertex = targetVertexMaybe.get();

            final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
            final Map<String, INDArray> targetParams = targetVertex.paramTable(false);

            TransferTask.ListBuilder taskBuilder = initReshapeListBuilder(registry, layerName, sourceParams, targetParams);

            transferOutputParameters(registry, newGraph, targetVertex, taskBuilder);

            taskBuilder.build().execute();
        }
    }

    private TransferTask.ListBuilder initReshapeListBuilder(
            TransferRegistry registry,
            String layerName,
            Map<String, INDArray> sourceParams,
            Map<String, INDArray> targetParams) {

        final SingleTransferTask.Builder firstTaskBuilder = SingleTransferTask.builder()
                .maskDim(1) // Dim 1 is input dim. Shall be transferred based on cropped outputs from previous layer
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(sourceParams.get(DefaultParamInitializer.WEIGHT_KEY), layerName + "_source_W"))
                        .build())
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(targetParams.get(DefaultParamInitializer.WEIGHT_KEY), layerName + "_target_W"))
                        .build());

        compFactory.apply(layerName)
                .ifPresent(firstTaskBuilder::compFactory);

        return firstTaskBuilder
                .addDependentTask(SingleTransferTask.builder()
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(sourceParams.get(DefaultParamInitializer.BIAS_KEY), layerName + "_source_b"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : 1)
                                .build())
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(targetParams.get(DefaultParamInitializer.BIAS_KEY), layerName + "_target_b"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim)
                                .build())
                );
    }

    private void transferOutputParameters(
            TransferRegistry registry,
            ComputationGraph newGraph,
            GraphVertex rootNode,
            TransferTask.ListBuilder taskListBuilder) {

        Stream.of(Optional.ofNullable(rootNode.getOutputVertices()).orElse(new VertexIndices[0]))
                .map(vertexIndex -> newGraph.getVertices()[vertexIndex.getVertexIndex()])
                .forEachOrdered(vertex -> {
                    final String layerName = vertex.getVertexName();

                    Optional<GraphVertex> sourceVertexMaybe = findLayerVertex(layerName, graph);
                    Optional<GraphVertex> targetVertexMaybe = findLayerVertex(layerName, newGraph);

                    if (sourceVertexMaybe.isPresent() && targetVertexMaybe.isPresent()) {
                        final GraphVertex sourceVertex = sourceVertexMaybe.get();
                        final GraphVertex targetVertex = targetVertexMaybe.get();

                        final Map<String, INDArray> sourceParams = sourceVertex.paramTable(false);
                        final Map<String, INDArray> targetParams = targetVertex.paramTable(false);
                        final INDArray sourceParam = sourceParams.get(DefaultParamInitializer.WEIGHT_KEY);
                        final INDArray targetParam = targetParams.get(DefaultParamInitializer.WEIGHT_KEY);

                        taskListBuilder.addDependentTask(SingleTransferTask.builder()
                                .source(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(sourceParam, layerName + "_source_W"))
                                        .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim) // flip dim 0 and 1
                                        .build())
                                .target(SingleTransferTask.IndMapping.builder()
                                        .entry(registry.register(targetParam, layerName + "_target_W"))
                                        .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim) // flip dim 0 and 1
                                        .build()));
                    }
                    if (Mutation.doesNinPropagateToNext(vertex)) {
                        transferOutputParameters(registry, newGraph, vertex, taskListBuilder);
                    }
                });
    }
}
