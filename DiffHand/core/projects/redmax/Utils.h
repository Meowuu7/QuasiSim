#pragma once
#include "Common.h"

namespace redmax {

namespace constants {
    const Vector3 gravity_dir = -Vector3::UnitZ();
    const dtype pi = (dtype)acos(-1.);
    const dtype eps = (dtype)1e-8;
    const int num_tangent_basis = 4;
}

namespace math {
    const dtype eps = (dtype)1e-9;
    const dtype eps_big = (dtype)1e-5;

    inline dtype deg2rad(dtype deg) {
        return deg / (dtype)180.0 * constants::pi;
    }

    inline dtype rad2deg(dtype rad) {
        return rad / constants::pi * (dtype)180.0;
    }
    
    inline Matrix3 skew(Vector3 v) {
        Matrix3 res;
        res(0, 0) = 0; res(0, 1) = -v(2); res(0, 2) = v(1);
        res(1, 0) = v(2); res(1, 1) = 0; res(1, 2) = -v(0);
        res(2, 0) = -v(1); res(2, 1) = v(0); res(2, 2) = 0;
        return res;
    }

    inline Vector3 unskew(Matrix3 a) {
        Vector3 res;
        res(0) = a(2, 1);
        res(1) = a(0, 2);
        res(2) = a(1, 0);
        return res;
    }

    inline SE3 SE(Matrix3 R, Vector3 p) {
        SE3 E = SE3::Identity();
        E.topLeftCorner(3, 3).noalias() = R;
        E.topRightCorner(3, 1).noalias() = p;
        return E;
    }

    inline SE3 Einv(SE3 E) {
        Matrix3 R_trans;
        R_trans.noalias() = E.topLeftCorner(3, 3).transpose();
        SE3 E_inv;
        E_inv.setIdentity();
        E_inv.topLeftCorner(3, 3) = R_trans;
        E_inv.topRightCorner(3, 1) = -R_trans * E.topRightCorner(3, 1);

        return E_inv;
    }

    inline Matrix6 Ad(SE3 E) {
        Matrix3 R = E.topLeftCorner(3, 3);
        Matrix6 Ad;
        Ad.setZero();
        Ad.topLeftCorner(3, 3) = R;
        Ad.bottomLeftCorner(3, 3) = skew(E.topRightCorner(3, 1)) * R;
        Ad.bottomRightCorner(3, 3) = R;

        return Ad;
    }

    inline Matrix6 ad(se3 phi) {
        Matrix3 skew_w = skew(phi.head(3));
        Matrix3 skew_v = skew(phi.tail(3));
        Matrix6 ad;
        ad.setZero();
        ad.topLeftCorner(3, 3) = skew_w;
        ad.bottomLeftCorner(3, 3) = skew_v;
        ad.bottomRightCorner(3, 3) = skew_w;

        return ad;
    }

    inline Matrix3 exp(Vector3 w) {
        Matrix3 I = Matrix3::Identity();
        dtype wlen = w.norm();
        Matrix3 R = I;
        if (wlen > 1e-12) {
            w = w / wlen;
            dtype wx = w(0);
            dtype wy = w(1);
            dtype wz = w(2);
            dtype c = cos(wlen), s = sin(wlen);
            dtype c1 = (dtype)1.0 - c;
            R(0, 0) = c + wx * wx * c1; R(0, 1) = -wz * s + wx * wy * c1; R(0, 2) = wy * s + wx * wz * c1;
            R(1, 0) = wz * s + wx * wy * c1; R(1, 1) = c + wy * wy * c1; R(1, 2) = -wx * s + wy * wz * c1;
            R(2, 0) = -wy * s + wx * wz * c1; R(2, 1) = wx * s + wy * wz * c1; R(2, 2) = c + wz * wz * c1;
        }
        return R;
    }

    inline Matrix4 exp(Vector6 q) {
        Matrix3 I = Matrix3::Identity();
        Vector3 w = q.head(3);
        dtype wlen = w.norm();
        Matrix3 R = I;
        if (wlen > 1e-12) {
            w = w / wlen;
            dtype wx = w(0);
            dtype wy = w(1);
            dtype wz = w(2);
            dtype c = cos(wlen), s = sin(wlen);
            dtype c1 = (dtype)1.0 - c;
            R(0, 0) = c + wx * wx * c1; R(0, 1) = -wz * s + wx * wy * c1; R(0, 2) = wy * s + wx * wz * c1;
            R(1, 0) = wz * s + wx * wy * c1; R(1, 1) = c + wy * wy * c1; R(1, 2) = -wx * s + wy * wz * c1;
            R(2, 0) = -wy * s + wx * wz * c1; R(2, 1) = wx * s + wy * wz * c1; R(2, 2) = c + wz * wz * c1;
        }

        Matrix4 E = Matrix4::Identity();
        E.topLeftCorner(3, 3) = R;
        Vector3 v = q.tail(3);
        if (wlen > 1e-12) {
            v = v / wlen;
            Matrix3 A = I - R;
            Vector3 cc = w.cross(v);
            Vector3 d = A * cc;
            dtype wv = w.dot(v);
            Vector3 p = (wv * wlen) * w + d;
            E.topRightCorner(3, 1) = p;
        } else {
            E.topRightCorner(3, 1) = v;
        }

        return E;
    }

    inline MatrixX gamma(Vector3 xi) {
        MatrixX G(3, 6);
        G.leftCols(3) = skew(xi).transpose();
        G.rightCols(3).setIdentity();
        return G;
    }

    inline Matrix3 quat2mat(Vector4 quat) {
        quat.normalize();
        dtype w = quat(0), x = quat(1), y = quat(2), z = quat(3);
        // Matrix3 mat = (Matrix3() << 1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
        //                         2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
        //                         2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y).finished();
        // return mat;
        return Eigen::Quaternion<dtype>(w, x, y, z).toRotationMatrix();
    }

    inline Vector4 mat2quat(Matrix3 mat) {
        // reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        Vector4 quat;
        dtype trace = mat.trace();
        if( trace > 0 ) {// I changed M_EPSILON to 0
            dtype s = 0.5 / sqrt(trace + 1.0);
            quat(0) = 0.25 / s;
            quat(1) = (mat(2, 1) - mat(1, 2)) * s;
            quat(2) = (mat(0, 2) - mat(2, 0)) * s;
            quat(3) = (mat(1, 0) - mat(0, 1)) * s;
        } else {
            if (mat(0, 0) > mat(1, 1) && mat(0, 0) > mat(2, 2)) {
                dtype s = 2.0 * sqrt(1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2));
                quat(0) = (mat(2, 1) - mat(1, 2)) / s;
                quat(1) = 0.25 * s;
                quat(2) = (mat(0, 1) + mat(1, 0)) / s;
                quat(3) = (mat(0, 2) + mat(2, 0)) / s;
                } else if (mat(1, 1) > mat(2, 2)) {
                    dtype s = 2.0 * sqrt(1.0f + mat(1, 1) - mat(0, 0) - mat(2, 2));
                    quat(0) = (mat(0, 2) - mat(2, 0)) / s;
                    quat(1) = (mat(0, 1) + mat(1, 0)) / s;
                    quat(2) = 0.25 * s;
                    quat(3) = (mat(1, 2) + mat(2, 1)) / s;
                } else {
                    float s = 2.0 * sqrt(1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1));
                    quat(0) = (mat(1, 0) - mat(0, 1)) / s;
                    quat(1) = (mat(0, 2) + mat(2, 0)) / s;
                    quat(2) = (mat(1, 2) + mat(2, 1)) / s;
                    quat(3) = 0.25 * s;
                }
        }
        return quat;
    }

    inline VectorX flatten_R(Matrix3 R) {
        VectorX flat_R = VectorX::Zero(9);
        for (int i = 0;i < 3;i++)
            for (int j = 0;j < 3;j++)
                flat_R(i * 3 + j) = R(i, j);
        return flat_R;
    }

    inline VectorX flatten_E(Matrix4 E) {
        VectorX flat_E = VectorX::Zero(12);
        flat_E.head(9) = flatten_R(E.topLeftCorner(3, 3));
        flat_E.tail(3) = E.topRightCorner(3, 1);

        return flat_E;
    }

    inline Matrix3 compose_R(VectorX flat_R) {
        Matrix3 R;
        for (int i = 0;i < 3;i++)
            for (int j = 0;j < 3;j++)
                R(i, j) = flat_R(i * 3 + j);
        return R;
    }

    inline Matrix4 compose_E(VectorX flat_E) {
        Matrix4 E = Matrix4::Identity();
        E.topLeftCorner(3, 3) = compose_R(flat_E.head(9));
        E.topRightCorner(3, 1) = flat_E.tail(3);
        return E;
    }
}

class SparseJacobianMatrixVector {
public:
    std::vector<SparseMatrixX> _data;

    SparseJacobianMatrixVector() {}
    SparseJacobianMatrixVector(int mat_rows, int mat_cols, int vec_dim) {
        if (mat_rows > 0 && mat_cols > 0 && vec_dim > 0)
            _data = std::vector<SparseMatrixX>(vec_dim, SparseMatrixX(mat_rows, mat_cols));
    }

    SparseMatrixX& operator()(int idx) {
        return _data[idx];
    }

    friend ostream& operator << (ostream& os, const SparseJacobianMatrixVector& jacobian) {
        for (auto i = 0;i < jacobian._data.size();i++) {
            os << "(" << i << ")" << std::endl << jacobian._data[i] << std::endl;
        }
        return os;
    }

    void setZero() {
        for (int i = 0;i < _data.size();i++) {
            _data[i].setZero();
        }
    }

    void setFromTriplets(std::vector<std::vector<TripletX>>& triplets) {
        for (int i = 0;i < _data.size();i++) {
            _data[i].setFromTriplets(triplets[i].begin(), triplets[i].end());
        }
    }

    std::size_t size() {
        return _data.size();
    }
};

class JacobianMatrixVector {
public:
    std::vector<MatrixX> _data;

    JacobianMatrixVector() {}
    JacobianMatrixVector(int mat_rows, int mat_cols, int vec_dim) {
        if (mat_rows > 0 && mat_cols > 0 && vec_dim > 0)
            _data = std::vector<MatrixX>(vec_dim, MatrixX::Zero(mat_rows, mat_cols));
    }

    MatrixX& operator()(int idx) {
        return _data[idx];
    }

    friend ostream& operator << (ostream& os, const JacobianMatrixVector& jacobian) {
        for (auto i = 0;i < jacobian._data.size();i++) {
            os << "(" << i << ")" << std::endl << jacobian._data[i] << std::endl;
        }
        return os;
    }

    void setZero() {
        for (int i = 0;i < _data.size();i++) {
            _data[i].setZero();
        }
    }

    SparseJacobianMatrixVector sparseView() {
        SparseJacobianMatrixVector sparse(_data[0].rows(), _data[0].cols(), _data.size());
        for (int i = 0;i < _data.size();i++) {
            // sparse(i) = _data[i].sparseView(1e-12, 1.);
            sparse(i) = _data[i].sparseView();
        }
        
        return sparse;
    }

    std::size_t size() {
        return _data.size();
    }
};

// inline void print_error(std::string name, VectorX a, VectorX b) {
//     dtype error = (a - b).norm();
//     dtype norm_a = a.norm();
//     dtype norm_b = b.norm();
//     dtype error_rel = error / max(min(norm_a, norm_b), 1e-4);
//     printf("%s: error = %.8lf, error_rel = %.8lf\n", name.c_str(), error, error_rel);
// }

inline void print_error(std::string name, dtype a, dtype b) {
    dtype error = fabs(a - b);
    dtype norm_a = fabs(a);
    dtype norm_b = fabs(b);
    dtype error_rel = error / max(min(norm_a, norm_b), (dtype)1e-4);
    if (error_rel > 1e-5 && error > 1e-5)
        printf("%s: error = %.10lf, error_rel = %.10lf\n", name.c_str(), error, error_rel);
}

inline void print_error(std::string name, MatrixX a, MatrixX b) {
    dtype error = (a - b).norm();
    dtype norm_a = a.norm();
    dtype norm_b = b.norm();
    dtype error_rel = error / max(min(norm_a, norm_b), (dtype)1e-4);
    if (error_rel > 1e-5 && error > 1e-5) {
        printf("%s: error = %.10lf, error_rel = %.10lf\n", name.c_str(), error, error_rel);
    }
}

inline void print_error(std::string name, JacobianMatrixVector a, JacobianMatrixVector b) {
    std::size_t n = a.size();
    dtype error_sq = 0.;
    dtype norm_sq_a = 0.;
    dtype norm_sq_b = 0.;
    for (auto i = 0;i < n;i++) {
        dtype error = (a(i) - b(i)).norm();
        dtype norm_a = a(i).norm();
        dtype norm_b = b(i).norm();
        error_sq += error * error;
        norm_sq_a += norm_a * norm_a;
        norm_sq_b += norm_b * norm_b;
    }
    dtype error = sqrt(error_sq);
    dtype norm_a = sqrt(norm_sq_a);
    dtype norm_b = sqrt(norm_sq_b);
    dtype error_rel = error / max(min(norm_a, norm_b), (dtype)1e-4);
    if (error_rel > 1e-5 && error > 1e-5) {
        printf("%s: error = %.10lf, error_rel = %.10lf\n", name.c_str(), error, error_rel);
    }
}

inline void print_error_full(std::string name, MatrixX a, MatrixX b) {
    dtype error = (a - b).norm();
    dtype norm_a = a.norm();
    dtype norm_b = b.norm();
    dtype error_rel = error / max(min(norm_a, norm_b), (dtype)1e-4);
    printf("%s: error = %.10lf, error_rel = %.10lf\n", name.c_str(), error, error_rel);
}

inline void print_error_full(std::string name, JacobianMatrixVector a, JacobianMatrixVector b) {
    std::size_t n = a.size();
    dtype error_sq = 0.;
    dtype norm_sq_a = 0.;
    dtype norm_sq_b = 0.;
    for (auto i = 0;i < n;i++) {
        dtype error = (a(i) - b(i)).norm();
        dtype norm_a = a(i).norm();
        dtype norm_b = b(i).norm();
        error_sq += error * error;
        norm_sq_a += norm_a * norm_a;
        norm_sq_b += norm_b * norm_b;
    }
    dtype error = sqrt(error_sq);
    dtype norm_a = sqrt(norm_sq_a);
    dtype norm_b = sqrt(norm_sq_b);
    dtype error_rel = error / max(min(norm_a, norm_b), (dtype)1e-4);
    printf("%s: error = %.10lf, error_rel = %.10lf\n", name.c_str(), error, error_rel);
}

inline std::string directory_of(const std::string& fname)
{
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}

// map value from [min1, max1] to [min2, max2]
inline VectorX map_value(const VectorX& value, const VectorX& min1, const VectorX& max1, const VectorX& min2, const VectorX& max2) {
    return (value - min1).cwiseProduct((max1 - min1).cwiseInverse()).cwiseProduct(max2 - min2) + min2;
}

inline VectorX str_to_eigen(std::string str) {
    std::stringstream iss(str);

    dtype value;
    std::vector<dtype> values;
    while (iss >> value) {
        values.push_back(value);
    }

    VectorX vec(values.size());
    for (int i = 0;i < values.size();i++)
        vec(i) = values[i];
    
    return vec;
}

inline VectorXi str_to_eigen_int(std::string str) {
    std::stringstream iss(str);

    int value;
    std::vector<int> values;
    while (iss >> value) {
        values.push_back(value);
    }

    VectorXi vec(values.size());
    for (int i = 0;i < values.size();i++)
        vec(i) = values[i];
    
    return vec;
}

inline void merge_meshes(std::vector<Matrix3Xf> &vertices, std::vector<Matrix3Xi> &faces, 
                            Matrix3Xf &merged_vertices, Matrix3Xi & merged_faces) {
    
    assert(vertices.size() == faces.size());
    
    int total_vertices = 0, total_faces = 0;
    for (int i = 0;i < vertices.size();++i) {
        total_vertices += vertices[i].cols();
        total_faces += faces[i].cols();
    }

    merged_vertices.resize(3, total_vertices);
    merged_faces.resize(3, total_faces);

    int vertices_offset = 0, faces_offset = 0;
    for (int i = 0;i < vertices.size();++i) {
        for (int j = 0;j < vertices[i].cols();++j)
            merged_vertices.col(vertices_offset + j) = vertices[i].col(j);
        for (int j = 0;j < faces[i].cols();++j)
            merged_faces.col(faces_offset + j) = faces[i].col(j) + Vector3i::Constant(vertices_offset);
        vertices_offset += vertices[i].cols();
        faces_offset += faces[i].cols();
    }
}

}