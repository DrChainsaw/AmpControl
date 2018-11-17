package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.GraphUtils;
import ampcontrol.model.training.model.evolve.mutate.util.TraverseBuilder;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;

import java.util.AbstractMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ParameterTransfer} with
 *
 * @author Christian Skärby
 */
public class ParameterTransferCrossover {

    @Test
    public void crossSimpleConv() {
        crossover(
                InputType.convolutional(33, 33, 3),
                GraphUtils.getCnnGraph("first1", "first2", "first3"),
                "first2",
                GraphUtils.getCnnGraph("second1", "second2", "second3"),
                "second1"
        );
    }

    private static void crossover(
            InputType inputType,
            ComputationGraph graph1,
            String name1,
            ComputationGraph graph2,
            String name2) {
        final VertexData vertex1 = createVertexData(name1, graph1, inputType);
        final VertexData vertex2 = createVertexData(name2, graph2, inputType);
        final GraphInfo result = new CrossoverPoint(vertex1, vertex2).execute();

        final ComputationGraph crossoverGraph = new ComputationGraph(result
                .builder()
                .build());
        crossoverGraph.init();

        Map<String, GraphVertex> vertexToGraph =
                Stream.concat(
                        result.verticesFrom(vertex1.info())
                                .map(nameMapping -> new AbstractMap.SimpleEntry<>(nameMapping.getNewName(), graph1.getVertex(nameMapping.getOldName()))),
                        result.verticesFrom(vertex2.info())
                                .map(nameMapping -> new AbstractMap.SimpleEntry<>(nameMapping.getNewName(), graph2.getVertex(nameMapping.getOldName()))))
                        .collect(Collectors.toMap(
                                Map.Entry::getKey,
                                Map.Entry::getValue
                        ));

        Function<String, GraphVertex> graphFunction = vertexToGraph::get;
        final ComputationGraph newGraph = new ParameterTransfer(graphFunction).transferWeightsTo(crossoverGraph);

        final Set<String> testedLayers = new HashSet<>();

        Stream.of(newGraph.getVertices())
                .filter(GraphVertex::hasLayer)
                .map(GraphVertex::getLayer)
                .filter(layer -> layer.numParams() > 0)
                .filter(layer -> result.verticesFrom(vertex1.info())
                        .map(GraphInfo.NameMapping::getNewName)
                        .anyMatch(vertex -> layer.conf().getLayer().getLayerName().equals(vertex)))
                .peek(layer -> testedLayers.add(layer.conf().getLayer().getLayerName()))
                .forEach(layer ->
                        assertEquals("Weights not transferred to layer " + layer.conf().getLayer().getLayerName() + "!",
                                graph1.getLayer(layer.conf().getLayer().getLayerName()).params().meanNumber(),
                                layer.params().meanNumber())
                );

        Stream.of(newGraph.getVertices())
                .filter(GraphVertex::hasLayer)
                .map(GraphVertex::getLayer)
                .filter(layer -> layer.numParams() > 0)
                .filter(layer -> result.verticesFrom(vertex2.info())
                        .map(GraphInfo.NameMapping::getNewName)
                        .anyMatch(vertex -> layer.conf().getLayer().getLayerName().equals(vertex)))
                .peek(layer -> testedLayers.add(layer.conf().getLayer().getLayerName()))
                .filter(layer -> !layer.conf().getLayer().getLayerName().equals("second1"))// size changed, so below is not correct
                .filter(layer -> !layer.conf().getLayer().getLayerName().equals("batchNorm_0"))// same as above?
                .forEach(layer ->
                        assertEquals("Weights not transferred to layer " + layer.conf().getLayer().getLayerName() + "!",
                                graph2.getLayer(layer.conf().getLayer().getLayerName()).params().meanNumber(),
                                layer.params().meanNumber())
                );

        assertEquals("Incorrect tested layers!", testedLayers, TraverseBuilder.forwards(newGraph)
                .traverseCondition(vertex -> true)
                .build()
                .children(newGraph.getConfiguration().getNetworkInputs().get(0))
                .filter(vertex -> newGraph.getVertex(vertex).numParams() > 0)
                .collect(Collectors.toSet()));

    }

    private static VertexData createVertexData(String name, ComputationGraph graph, InputType inputType) {
        return new VertexData(name, new GraphInfo.Input(new ComputationGraphConfiguration.GraphBuilder(
                graph.getConfiguration(), new NeuralNetConfiguration.Builder(graph.conf())
                .weightInit(WeightInit.ZERO)
        )
                .setInputTypes(inputType)));

    }
}
