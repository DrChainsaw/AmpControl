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

    /**
     * Test parameter transfer when two simple cnns are crossed
     */
    @Test
    public void crossSimpleConv() {
        final String[] first = {"first1", "first2", "first3"};
        final String[] second = {"second1", "second2", "second3"};
        final ComputationGraph graphFirst = GraphUtils.getCnnGraph(first[0], first[1], first[2]);
        final ComputationGraph graphSecond = GraphUtils.getCnnGraph(second[0], second[1], second[2]);

        for (String firstCp : first) {
            for (String secondCp : second) {
                crossover(
                        InputType.convolutional(33, 33, 3),
                        graphFirst,
                        firstCp,
                        graphSecond,
                        secondCp
                );
            }
        }
    }

    /**
     * Test parameter transfer when two simple dense networks are crossed
     */
    @Test
    public void crossSimpleDense() {
        final String[] first = {"first1", "first2", "first3"};
        final String[] second = {"second1", "second2", "second3"};
        final ComputationGraph graphFirst = GraphUtils.getGraph(first[0], first[1], first[2]);
        final ComputationGraph graphSecond = GraphUtils.getGraph(second[0], second[1], second[2]);

        for (String firstCp : first) {
            for (String secondCp : second) {
                crossover(
                        InputType.feedForward(33),
                        graphFirst,
                        firstCp,
                        graphSecond,
                        secondCp
                );
            }
        }
    }

    /**
     * Test parameter transfer when a residual net and a fork net are crossed
     */
    @Test
    public void crossResNetAndForkNet() {
        final String[] first = {"first1", "first2", "first3"};
        final String[] second = {"second1", "second2"};
        final ComputationGraph graphFirst = GraphUtils.getResNet(first[0], first[1], first[2]);
        final ComputationGraph graphSecond = GraphUtils.getForkNet(second[0], second[1], "secFork1", "secFork2", "secFork3");

        for (String firstCp : first) {
            for (String secondCp : second) {
                crossover(
                        InputType.convolutional(33, 33, 3),
                        graphFirst,
                        firstCp,
                        graphSecond,
                        secondCp
                );
            }
        }
    }

    /**
     * Test parameter transfer when a fork net and a forked residual net are crossed
     */
    @Test
    public void crossForkNetAndResFork() {
        final String[] first = {"first1", "first2"};
        final String[] second = {"second1", "second2"};
        final ComputationGraph graphFirst = GraphUtils.getForkResNet(first[0], first[1], "firFork1", "firFork2");
        final ComputationGraph graphSecond = GraphUtils.getForkNet(second[0], second[1], "secFork1", "secFork2", "secFork3");

        for (String firstCp : first) {
            for (String secondCp : second) {
                crossover(
                        InputType.convolutional(33, 33, 3),
                        graphFirst,
                        firstCp,
                        graphSecond,
                        secondCp
                );
            }
        }
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

        final Map<String, GraphVertex> nameToVertex =
                Stream.concat(
                        result.verticesFrom(vertex1.info())
                                .map(nameMapping -> new AbstractMap.SimpleEntry<>(nameMapping.getNewName(), graph1.getVertex(nameMapping.getOldName()))),
                        result.verticesFrom(vertex2.info())
                                .map(nameMapping -> new AbstractMap.SimpleEntry<>(nameMapping.getNewName(), graph2.getVertex(nameMapping.getOldName()))))
                        .collect(Collectors.toMap(
                                Map.Entry::getKey,
                                Map.Entry::getValue
                        ));

        final Function<String, GraphVertex> nameToVertexFunction = nameToVertex::get;
        final ComputationGraph newGraph = new ParameterTransfer(nameToVertexFunction).transferWeightsTo(crossoverGraph);

        final Set<String> testedLayers = new HashSet<>();
        Stream.of(newGraph.getVertices())
                .filter(vertex -> vertex.numParams() > 0)
                .peek(vertex -> testedLayers.add(vertex.getVertexName())) // Comparison below does not work when nOut or nIn is changed
                .filter(vertex -> vertex.numParams() == nameToVertex.get(vertex.getVertexName()).numParams())
                .forEach(vertex ->
                        assertEquals("Weights not transferred to vertex " + vertex.getVertexName() + "!",
                                nameToVertex.get(vertex.getVertexName()).params().meanNumber(),
                                vertex.params().meanNumber())
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
