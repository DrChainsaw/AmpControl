package ampcontrol.model.training.model.description;

import org.junit.Test;

import java.util.ArrayList;

/**
 * Test cases for {@link MutatingConv2dFactory}
 *
 * @author Christian Skärby
 */
public class MutatingConv2dFactoryTest {

    /**
     * Just a smoke test...
     */
    @Test
    public void addModelData() {
        new FactoryInitializer()
                .initialize(
                        ((tr, ev, is, np, md) -> new MutatingConv2dFactory(tr,ev,is,np,md).addModelData(new ArrayList<>()))
                );
    }
}