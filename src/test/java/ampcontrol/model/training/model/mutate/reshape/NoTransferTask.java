package ampcontrol.model.training.model.mutate.reshape;

import java.util.Comparator;

/**
 * No-op used to terminate a chain of {@link ReshapeSubTask}s
 *
 * @author Christian Skärby
 */
public class NoTransferTask implements ReshapeSubTask {
    @Override
    public void addWantedElementsFromSource(int dim, int[] indexes) {
        // Ignore
    }

    @Override
    public void addWantedNrofElementsFromTarget(int dim, int nrofElements) {
        // Ignore
    }

    @Override
    public Comparator<Integer> getComparator(int[] tensorDimensions) {
        return null;
    }

    @Override
    public void execute() {
        // Ignore
    }
}
