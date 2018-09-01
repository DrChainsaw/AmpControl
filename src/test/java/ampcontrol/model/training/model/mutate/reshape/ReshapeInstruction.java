package ampcontrol.model.training.model.mutate.reshape;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder(builderClassName = "Builder")
public class ReshapeInstruction {
    private final long[] targetShape;
    private final long[] sourceShape;
    private final Prune.PruneInstruction pruneInstruction;

    public void reshape() {
        pruneInstruction.assign();
    }
}
