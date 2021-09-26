import getdata as gt
import network as nt


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = gt.read_img()
    nt.train_network(train_x, train_y, test_x, test_y, epoch=2, batch_size=16)
    nt.load_network_then_train(train_x, train_y, test_x, test_y, epoch=2, batch_size=16,
                               input_name='first_model.h5', output_name='second_model.h5')
    nt.plot_result(test_x, input_name='second_model.h5', index=0)
