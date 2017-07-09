
class EncoderModel:
    def features_get_from_image(self, img):
        print('should be overridden')
        assert False

    def features_get_from_path(self, path):
        print('should be overridden')
        assert False

    def features_get_from_images(self, imgs):
        print('should be overridden')
        assert False
    def image_from_features(self, features):
        print('should be overridden')
        assert False
