package ampcontrol.model.training.model.evolve.crossover.graph;

import ampcontrol.model.training.model.evolve.crossover.Crossover;
import ampcontrol.model.training.model.evolve.mutate.util.BackwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.ForwardOf;
import ampcontrol.model.training.model.evolve.mutate.util.Graph;
import ampcontrol.model.training.model.evolve.mutate.util.Traverse;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Comparator;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Simple {@link Crossover} for {@link GraphBuilder}s. To keep things reasonably simple, only vertices which are the
 * single vertex connecting all vertices above and below them are eligible for crossover.
 * <br><br>
 * Assumes the input types have been set for the inputs.
 *
 * @author Christian Skärby
 */
public final class SinglePoint implements Crossover<GraphInfo> {

    private static final Logger log = LoggerFactory.getLogger(CrossoverPoint.class);

    private final Supplier<PointSelection> selectionSupplier;

    public static class PointSelection {
        private final double relativeLocation;
        private final double distance;

        public PointSelection(double distance, double relativeLocation) {
            this.relativeLocation = relativeLocation;
            this.distance = distance;
        }

        private double locationTarget() {
            return relativeLocation;
        }

        private double distanceTarget() {
            return distance;
        }
    }

    public SinglePoint(Supplier<PointSelection> selectionSupplier) {
        this.selectionSupplier = selectionSupplier;
    }

    @Override
    public GraphInfo cross(
            GraphInfo bottom,
            GraphInfo top) {

        final PointSelection pointSelection = selectionSupplier.get();

        log.info("Crossover target distance: " + pointSelection.distanceTarget() + " location: " + pointSelection.locationTarget());

        final Comparator<CrossoverPoint> byDistance = Comparator.comparingDouble(
                point -> Math.abs(pointSelection.distanceTarget() - point.distance()));
        final Comparator<CrossoverPoint> byBottomLocation = Comparator.comparingDouble(
                point -> Math.abs(pointSelection.locationTarget() - point.bottom().location()));
        final Comparator<CrossoverPoint> byTiebreaker = Comparator.comparingInt(point -> (point.bottom().name() + point.top().name()).hashCode());

        final Graph<String> childrenBottom = new ForwardOf(bottom.builder());
        final Graph<String> parentsTop = new BackwardOf(top.builder());

        return findValidVertices(bottom)
                // Don't allow bottom vertex if it has an EpsilonSpyVertex as a child as it might (and will) lead to
                // non-weight vertices being spyed on
                .filter(bottomVertex -> childrenBottom.children(bottomVertex.name())
                        .map(childName -> bottom.builder().getVertices().get(childName))
                        .noneMatch(childVertex -> childVertex instanceof EpsilonSpyVertex))
                .flatMap(bottomVertex -> findValidVertices(top)
                       // .filter(topVertex -> topVertex.type() == bottomVertex.type())
                       // .filter(topVertex -> isShapesSafe(topVertex.shape(), bottomVertex.shape()))
                        .filter(topVertex -> parentsTop.children(topVertex.name())
                                .allMatch(childName -> new VertexData(childName, top).type() == bottomVertex.type()))
                        .filter(topVertex -> parentsTop.children(topVertex.name())
                                .allMatch(childName -> isShapesSafe(new VertexData(childName, top).shape(), bottomVertex.shape())))
                        .filter(topVertex -> !(top.builder().getVertices().get(topVertex.name()) instanceof EpsilonSpyVertex))
                        .map(topVertex -> new CrossoverPoint(bottomVertex, topVertex)))
                .min(byDistance.thenComparing(byBottomLocation).thenComparing(byTiebreaker))
                .map(CrossoverPoint::execute)
                .orElse(new GraphInfo.NoopResult(bottom)); // Not a single valid crossover point found
    }

    private static Stream<VertexData> findValidVertices(GraphInfo info) {
        return info.builder().getVertices().keySet().stream()
                .filter(vertex -> isValid(vertex, info.builder()))
                .map(vertex -> new VertexData(vertex, info));

    }

    private static boolean isValid(String vertex, GraphBuilder builder) {
        // Vertices could perhaps work, but would just be a waste as they will typically only be connected to one input
        if (
                builder.getVertices().get(vertex) instanceof ElementWiseVertex
                        || builder.getVertices().get(vertex) instanceof MergeVertex
                        || builder.getNetworkOutputs().contains(vertex)
                        || builder.getNetworkInputs().contains(vertex)) {
            return false;
        }

        if (new ForwardOf(builder).children(vertex)
                .map(childName -> builder.getVertices().get(childName))
                .anyMatch(childVertex -> childVertex instanceof EpsilonSpyVertex)) {
            return false;
        }

        final List<String> flattened = new Traverse<>(
                new ForwardOf(builder))
                .children(vertex)
                .collect(Collectors.toList());
        flattened.add(vertex);
        //Vertex is not inside any type of skip connection if we find all inputs to each found vertex in the flattened graph
        return flattened.stream()
                .filter(vert -> !vert.equals(vertex))
                .allMatch(vert -> flattened.containsAll(builder.getVertexInputs().get(vert)));
    }

    private static boolean isShapesSafe(long[] shapeBottom, long[] shapeTop) {
        return IntStream.range(1, shapeBottom.length)
                //.peek(dim -> System.out.println("size " + dim + ": " + shapeBottom[dim] + " vs " + shapeTop[dim]))
                .mapToDouble(dim -> (shapeBottom[dim] - shapeTop[dim]) / (double) (shapeBottom[dim] + shapeTop[dim]))
                //.peek(relSize -> System.out.println("relsize: " + relSize))
                .allMatch(relativeSize -> relativeSize > -0.1);

    }
}
