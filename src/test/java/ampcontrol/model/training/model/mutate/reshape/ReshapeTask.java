package ampcontrol.model.training.model.mutate.reshape;

import lombok.Builder;
import lombok.Getter;

import java.util.Comparator;
import java.util.Set;
import java.util.stream.IntStream;

@Getter
@Builder(builderClassName = "Builder")
public class ReshapeTask {
    private final long[] targetShape;
    private final long[] sourceShape;
    private final ReshapeSubTask reshapeSubTask;
    @lombok.Singular("maskDim") private final Set<Integer> dimensionMask;

    public void reshape() {
        for (int i = 0; i < targetShape.length; i++) {
            if (targetShape[i] < sourceShape[i] && !dimensionMask.contains(i)) {
                final int tensorDim = i;
                final int[] tensorDimensions = IntStream.range(0, targetShape.length)
                        .filter(dim -> tensorDim != dim)
                        .toArray();

                final Comparator<Integer> comparator = reshapeSubTask.getComparator(tensorDimensions);
                final int[] wantedElements = IntStream.range(0, (int) sourceShape[tensorDim])
                        .boxed()
                        .sorted(comparator)
                        .limit(targetShape[tensorDim])
                        .mapToInt(e -> e)
                        .sorted()
                        .toArray();

                reshapeSubTask.addWantedElements(tensorDim, wantedElements);
            }
        }

        reshapeSubTask.assign();
    }
}
