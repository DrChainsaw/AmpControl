package ampcontrol.model.training.model.mutate.reshape;

import lombok.Builder;
import lombok.Singular;

import java.util.Comparator;
import java.util.List;

@Builder
public class ReshapeSubTaskList implements ReshapeSubTask {
    @Singular
    private final List<ReshapeSubTask> instructions;

    @Override
    public void addWantedElements(int dim, int[] wantedElementInds) {
        instructions.forEach(instr -> instr.addWantedElements(dim, wantedElementInds));
    }

    @Override
    public Comparator<Integer> getComparator(int[] tensorDimensions) {
        return instructions.get(0).getComparator(tensorDimensions);
    }

    @Override
    public void assign() {
        instructions.forEach(ReshapeSubTask::assign);
    }
}
