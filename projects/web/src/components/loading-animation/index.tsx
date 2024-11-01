import styles from './loadingAnimation.module.scss';

interface ILoadingAnimationProps {
  className?: string;
}

const LoadingAnimation = (props: ILoadingAnimationProps) => {
  const { className } = props;
  return <div className={`${styles.loader} ${className}`}></div>;
};

export default LoadingAnimation;
