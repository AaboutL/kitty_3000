//
// Created by slam on 18-4-20.
//

#ifndef ALIGNMENT_3000FPS_CV_FACEMARK_H
#define ALIGNMENT_3000FPS_CV_FACEMARK_H

#include <opencv2/core.hpp>
#include <vector>


namespace ext {

/** @brief Abstract base class for all facemark models
@code
// Using Facemark in your code:
Ptr<Facemark> facemark = createFacemarkLBF();
@endcode
*/
    class Facemark {
    public:
        Facemark(){};
        virtual ~Facemark(){};

        /** @brief Clears the algorithm state
        */
        virtual void clear() {}

        /** @brief Stores algorithm parameters in a file storage
        */
        virtual void write(cv::FileStorage& fs) const { (void)fs; }

        /** @brief simplified API for language bindings
         * @overload
         */
        void write(const cv::Ptr<cv::FileStorage>& fs, const cv::String& name = cv::String()) const;

        /** @brief Reads algorithm parameters from a file storage
        */
        virtual void read(const cv::FileNode& fn) { (void)fn; }

        /** @brief Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
         */
        virtual bool empty() const { return false; }

        /** @brief Reads algorithm from the file node

         This is static template method of Algorithm. It's usage is following (in the case of SVM):
         @code
         cv::FileStorage fsRead("example.xml", FileStorage::READ);
         Ptr<SVM> svm = Algorithm::read<SVM>(fsRead.root());
         @endcode
         In order to make this method work, the derived class must overwrite Algorithm::read(const
         FileNode& fn) and also have static create() method without parameters
         (or with all the optional parameters)
         */
        template<typename _Tp> static cv::Ptr<_Tp> read(const cv::FileNode& fn)
        {
            cv::Ptr<_Tp> obj = _Tp::create();
            obj->read(fn);
            return !obj->empty() ? obj : cv::Ptr<_Tp>();
        }

        /** @brief Loads algorithm from the file

         @param filename Name of the file to read.
         @param objname The optional name of the node to read (if empty, the first top-level node will be used)

         This is static template method of Algorithm. It's usage is following (in the case of SVM):
         @code
         Ptr<SVM> svm = Algorithm::load<SVM>("my_svm_model.xml");
         @endcode
         In order to make this method work, the derived class must overwrite Algorithm::read(const
         FileNode& fn).
         */
        template<typename _Tp> static cv::Ptr<_Tp> load(const cv::String& filename, const cv::String& objname=cv::String())
        {
            cv::FileStorage fs(filename, cv::FileStorage::READ);
//            cv::CV_Assert(fs.isOpened());
            assert(fs.isOpened());
            cv::FileNode fn = objname.empty() ? fs.getFirstTopLevelNode() : fs[objname];
            if (fn.empty()) return cv::Ptr<_Tp>();
            cv::Ptr<_Tp> obj = _Tp::create();
            obj->read(fn);
            return !obj->empty() ? obj : cv::Ptr<_Tp>();
        }

        /** @brief Loads algorithm from a String

         @param strModel The string variable containing the model you want to load.
         @param objname The optional name of the node to read (if empty, the first top-level node will be used)

         This is static template method of Algorithm. It's usage is following (in the case of SVM):
         @code
         Ptr<SVM> svm = Algorithm::loadFromString<SVM>(myStringModel);
         @endcode
         */
        template<typename _Tp> static cv::Ptr<_Tp> loadFromString(const cv::String& strModel, const cv::String& objname=cv::String())
        {
            cv::FileStorage fs(strModel, cv::FileStorage::READ + cv::FileStorage::MEMORY);
            cv::FileNode fn = objname.empty() ? fs.getFirstTopLevelNode() : fs[objname];
            cv::Ptr<_Tp> obj = _Tp::create();
            obj->read(fn);
            return !obj->empty() ? obj : cv::Ptr<_Tp>();
        }

        /** Saves the algorithm to a file.
         In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs). */
//        virtual void save(const cv::String& filename) const;

        /** Returns the algorithm string identifier.
         This string is used as top level xml/yml node tag when the object is saved to a file or string. */
//        virtual cv::String getDefaultName() const;

    protected:
        void writeFormat(cv::FileStorage& fs) const;

    public:
        /** @brief A function to load the trained model before the fitting process.
        @param model A string represent the filename of a trained model.

        <B>Example of usage</B>
        @code
        facemark->loadModel("../data/lbf.model");
        @endcode
        */
        virtual void loadModel(cv::String model) = 0;
        // virtual void saveModel(String fs)=0;

        /** @brief Detect facial landmarks from an image.
        @param image Input image.
        @param faces Output of the function which represent region of interest of the detected faces.
        Each face is stored in cv::Rect container.
        @param landmarks The detected landmark points for each faces.

        <B>Example of usage</B>
        @code
        Mat image = imread("image.jpg");
        std::vector<Rect> faces;
        std::vector<std::vector<Point2f> > landmarks;
        facemark->fit(image, faces, landmarks);
        @endcode
        */
        virtual bool fit(cv::InputArray image,
                                 cv::InputArray faces,
                                 cv::OutputArrayOfArrays landmarks) = 0;
    }; /* Facemark*/


//! construct an AAM facemark detector
    cv::Ptr<Facemark> createFacemarkAAM();

//! construct an LBF facemark detector
    cv::Ptr<Facemark> createFacemarkLBF();

//! construct a Kazemi facemark detector
    cv::Ptr<Facemark> createFacemarkKazemi();


} // ext
#endif //ALIGNMENT_3000FPS_CV_FACEMARK_H
