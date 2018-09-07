package ampcontrol.model.training.model.evolve.transfer;

/**
 * No-op typically used to terminate a chain of dependent {@link TransferTask}s
 *
 * @author Christian Skärby
 */
public class NoTransferTask implements TransferTask {

    @Override
    public void addWantedElementsFromSource(int dim, int[] indexes) {
        // Ignore
    }

    @Override
    public void addWantedNrofElementsFromTarget(int dim, int nrofElements) {
        // Ignore
    }

    @Override
    public void execute() {
        // Ignore
    }
}
