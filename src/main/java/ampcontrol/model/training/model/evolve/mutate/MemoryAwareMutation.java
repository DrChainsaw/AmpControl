package ampcontrol.model.training.model.evolve.mutate;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.function.Function;

/**
 * Selects one out of two {@link Mutation}s to apply based on memory consumption provided by a {@link MemoryProvider}.
 *
 * @param <T>
 * @author Christian Skärby
 */
public class MemoryAwareMutation<T> implements Mutation<T> {

    /**
     * Interface to provide memory usage
     */
    interface MemoryProvider {
        double getUsage();
    }


    private static class DeviceMemoryProvider implements MemoryProvider {

        private final Pointer devicePointer;
        private final double totalMemory;

        private DeviceMemoryProvider() {
            this.devicePointer = new IntPointer(NativeOpsHolder.getInstance().getDeviceNativeOps().getDevice());
            this.totalMemory = NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceTotalMemory(devicePointer);
        }

        @Override
        public double getUsage() {
            return (totalMemory - NativeOpsHolder.getInstance().getDeviceNativeOps().getDeviceFreeMemory(devicePointer)) / totalMemory;
        }
    }

    private final MemoryProvider memoryProvider;
    private final Function<Double, Mutation<T>> mutationSelection;

    public MemoryAwareMutation(Function<Double, Mutation<T>> mutationSelection) {
        this(new DeviceMemoryProvider(), mutationSelection);
    }

    public MemoryAwareMutation(MemoryProvider memoryProvider, Function<Double, Mutation<T>> mutationSelection) {
        this.memoryProvider = memoryProvider;
        this.mutationSelection = mutationSelection;
    }

    @Override
    public T mutate(T toMutate) {
        return mutationSelection.apply(memoryProvider.getUsage()).mutate(toMutate);
    }
}
