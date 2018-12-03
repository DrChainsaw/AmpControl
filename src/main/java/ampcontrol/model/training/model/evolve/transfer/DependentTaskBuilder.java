package ampcontrol.model.training.model.evolve.transfer;

/**
 * Builder specialized in creating "optional" dependent tasks.
 *
 * @author Christian Skärby
 */
public class DependentTaskBuilder implements TransferTask.ListBuilder {

    private TransferTask.ListBuilder dependentTask = NoTransferTask.builder();

    @Override
    public TransferTask.ListBuilder addDependentTask(TransferTask.ListBuilder builder) {
        dependentTask = dependentTask.addDependentTask(builder);
        return this;
    }

    @Override
    public TransferTask build() {
        return dependentTask.build();
    }
}
