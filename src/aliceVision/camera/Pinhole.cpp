// This file is part of the AliceVision project.
// Copyright (c) 2023 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Pinhole.hpp"

namespace aliceVision {
namespace camera {

Mat3 Pinhole::K() const
{
    Mat3 K;

    Vec2 pp = getPrincipalPoint();

    K  << _scale(0), 0.0, pp(0),
            0.0, _scale(1), pp(1),
            0.0, 0.0, 1.0;
    return K;
}

void Pinhole::setK(double focalLengthPix, double ppx, double ppy)
{
    _scale(0) = focalLengthPix;
    _scale(1) = focalLengthPix;
    _offset(0) = ppx - static_cast<double>(_w) * 0.5;
    _offset(1) = ppy - static_cast<double>(_h) * 0.5;
}

void Pinhole::setK(const Mat3 & K)
{
    _scale(0) = K(0, 0);
    _scale(1) = K(1, 1);
    _offset(0) = K(0, 2) - static_cast<double>(_w) * 0.5;
    _offset(1) = K(1, 2) - static_cast<double>(_h) * 0.5;
}

Vec2 Pinhole::project(const geometry::Pose3& pose, const Vec4& pt, bool applyDistortion) const
{
    const Vec4 X = pose.getHomogeneous() * pt; // apply pose
    const Vec2 P = X.head<2>() / X(2);

    const Vec2 distorted = this->addDistortion(P);
    const Vec2 impt = this->cam2ima(distorted);

    return impt;
}

Eigen::Matrix<double, 2, 9> Pinhole::getDerivativeProjectWrtRotation(const geometry::Pose3& pose, const Vec4 & pt)
{
    const Vec4 X = pose.getHomogeneous() * pt; // apply pose

    const Eigen::Matrix<double, 3, 9> d_X_d_R = getJacobian_AB_wrt_A<3, 3, 1>(pose.rotation(), pt.head(3));

    const Vec2 P = X.head<2>() / X(2);

    Eigen::Matrix<double, 2, 3> d_P_d_X;
    d_P_d_X(0, 0) = 1 / X(2);
    d_P_d_X(0, 1) = 0;
    d_P_d_X(0, 2) = - X(0) / (X(2) * X(2));
    d_P_d_X(1, 0) = 0;
    d_P_d_X(1, 1) = 1 / X(2);
    d_P_d_X(1, 2) = - X(1) / (X(2) * X(2));

    return getDerivativeCam2ImaWrtPoint() * getDerivativeAddDistoWrtPt(P) * d_P_d_X * d_X_d_R;
}

Eigen::Matrix<double, 2, 16> Pinhole::getDerivativeProjectWrtPose(const geometry::Pose3& pose, const Vec4& pt) const
{
    const Eigen::Matrix4d T = pose.getHomogeneous();

    const Vec4 X = T * pt; // apply pose

    const Eigen::Matrix<double, 4, 16> d_X_d_T = getJacobian_AB_wrt_A<4, 4, 1>(T, pt);

    const Vec2 P = X.head<2>() / X(2);

    Eigen::Matrix<double, 2, 3> d_P_d_X;
    d_P_d_X(0, 0) = 1 / X(2);
    d_P_d_X(0, 1) = 0;
    d_P_d_X(0, 2) = - X(0) / (X(2) * X(2));
    d_P_d_X(1, 0) = 0;
    d_P_d_X(1, 1) = 1 / X(2);
    d_P_d_X(1, 2) = - X(1) / (X(2) * X(2));

    return getDerivativeCam2ImaWrtPoint() * getDerivativeAddDistoWrtPt(P) * d_P_d_X * d_X_d_T.block<3, 16>(0, 0);
}

Eigen::Matrix<double, 2, 4> Pinhole::getDerivativeProjectWrtPoint(const geometry::Pose3& pose, const Vec4 & pt) const
{
    const Eigen::Matrix4d T = pose.getHomogeneous();
    const Vec4 X = T * pt; // apply pose

    const Eigen::Matrix<double, 4, 4> & d_X_d_P = getJacobian_AB_wrt_B<4, 4, 1>(T, pt);

    const Vec2 P = X.head<2>() / X(2);

    Eigen::Matrix<double, 2, 4> d_P_d_X;
    d_P_d_X(0, 0) = 1 / X(2);
    d_P_d_X(0, 1) = 0;
    d_P_d_X(0, 2) = - X(0) / (X(2) * X(2));
    d_P_d_X(0, 3) = 0;
    d_P_d_X(1, 0) = 0;
    d_P_d_X(1, 1) = 1 / X(2);
    d_P_d_X(1, 2) = - X(1) / (X(2) * X(2));
    d_P_d_X(1, 3) = 0;

    return getDerivativeCam2ImaWrtPoint() * getDerivativeAddDistoWrtPt(P) * d_P_d_X * d_X_d_P;
}

Eigen::Matrix<double, 2, Eigen::Dynamic> Pinhole::getDerivativeProjectWrtDisto(const geometry::Pose3& pose, const Vec4 & pt) const
{
    const Vec4 X = pose.getHomogeneous() * pt; // apply pose
    const Vec2 P = X.head<2>() / X(2);

    return getDerivativeCam2ImaWrtPoint() * getDerivativeAddDistoWrtDisto(P);
}

Eigen::Matrix<double, 2, 2> Pinhole::getDerivativeProjectWrtPrincipalPoint(const geometry::Pose3& pose, const Vec4 & pt) const
{
    return getDerivativeCam2ImaWrtPrincipalPoint();
}

Eigen::Matrix<double, 2, 2> Pinhole::getDerivativeProjectWrtScale(const geometry::Pose3& pose, const Vec4 & pt) const
{
    const Vec4 X = pose.getHomogeneous() * pt; // apply pose
    const Vec2 P = X.head<2>() / X(2);

    const Vec2 distorted = this->addDistortion(P);

    return getDerivativeCam2ImaWrtScale(distorted);
}

Eigen::Matrix<double, 2, Eigen::Dynamic> Pinhole::getDerivativeProjectWrtParams(const geometry::Pose3& pose, const Vec4& pt3D) const
{
    Eigen::Matrix<double, 2, Eigen::Dynamic> ret(2, getParams().size());

    ret.block<2, 2>(0, 0) = getDerivativeProjectWrtScale(pose, pt3D);
    ret.block<2, 2>(0, 2) = getDerivativeProjectWrtPrincipalPoint(pose, pt3D);

    if (hasDistortion())
    {
        const size_t distortionSize = _pDistortion->getDistortionParametersCount();
        ret.block(0, 4, 2, distortionSize) = getDerivativeProjectWrtDisto(pose, pt3D);
    }

    return ret;
}

Vec3 Pinhole::toUnitSphere(const Vec2 & pt) const
{
    return pt.homogeneous().normalized();
}

Eigen::Matrix<double, 3, 2> Pinhole::getDerivativetoUnitSphereWrtPoint(const Vec2 & pt) const
{
    const double norm2 = pt(0)*pt(0) + pt(1)*pt(1) + 1.0;
    const double norm = sqrt(norm2);

    const Vec3 ptcam = pt.homogeneous();

    Eigen::Matrix<double, 1, 2> d_norm_d_pt;
    d_norm_d_pt(0, 0) = pt(0) / norm;
    d_norm_d_pt(0, 1) = pt(1) / norm;

    Eigen::Matrix<double, 3, 2> d_ptcam_d_pt;
    d_ptcam_d_pt(0, 0) = 1.0;
    d_ptcam_d_pt(0, 1) = 0.0;
    d_ptcam_d_pt(1, 0) = 0.0;
    d_ptcam_d_pt(1, 1) = 1.0;
    d_ptcam_d_pt(2, 0) = 0.0;
    d_ptcam_d_pt(2, 1) = 0.0;

    return (norm * d_ptcam_d_pt - ptcam * d_norm_d_pt) / norm2;
}

double Pinhole::imagePlaneToCameraPlaneError(double value) const
{
    return value / _scale(0);
}

Mat34 Pinhole::getProjectiveEquivalent(const geometry::Pose3 & pose) const
{
    Mat34 P;

    P_from_KRt(K(), pose.rotation(), pose.translation(), &P);
    return P;
}

bool Pinhole::isVisibleRay(const Vec3 & ray) const
{
    // if(ray(2) <= 0.0)
    if (ray(2) < std::numeric_limits<double>::epsilon())
    {
        return false;
    }

    const Vec2 proj = ray.head(2) / ray(2);

    const Vec2 p1 = removeDistortion(ima2cam(Vec2(0, 0)));
    const Vec2 p2 = removeDistortion(ima2cam(Vec2(_w, 0)));
    const Vec2 p3 = removeDistortion(ima2cam(Vec2(_w, _h)));
    const Vec2 p4 = removeDistortion(ima2cam(Vec2(0, _h)));

    const double xmin = std::min(p4(0), (std::min(p3(0), std::min(p1(0), p2(0)))));
    const double ymin = std::min(p4(1), (std::min(p3(1), std::min(p1(1), p2(1)))));
    const double xmax = std::max(p4(0), (std::max(p3(0), std::max(p1(0), p2(0)))));
    const double ymax = std::max(p4(1), (std::max(p3(1), std::max(p1(1), p2(1)))));

    if (proj(0) < xmin || proj(0) > xmax || proj(1) < ymin || proj(1) > ymax)
    {
        return false;
    }

    return true;
}

} // namespace camera
} // namespace aliceVision
